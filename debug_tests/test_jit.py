import ctypes
import os
from contextlib import nullcontext
from typing import Any, Dict

import cuda.bindings.driver as cbd
import torch

from deep_gemm import jit
from deep_gemm.cuda_utils import CALL_CUDA_FUNC, CUDA_SUCCESS
from deep_gemm.trace import create_tracer

# Essential debugging staffs
os.environ["DG_JIT_DEBUG"] = os.getenv("DG_JIT_DEBUG", "1")
os.environ["DG_JIT_DISABLE_CACHE"] = os.getenv("DG_JIT_DISABLE_CACHE", "1")
SHOULD_TRACE = os.getenv("SHOULD_TRACE", "0") == "1"


class VectorAddRuntime(jit.Runtime):
    def __init__(self, path: str) -> None:
        super().__init__(path)

    @staticmethod
    def generate(kwargs: Dict[str, Any]) -> str:
        return f"""
#ifdef __CUDACC_RTC__
#include <deep_gemm/nvrtc_std.cuh>
#else
#include <cuda.h>
#endif

#include <cuda_fp8.h>
#include <cuda_bf16.h>

template <typename T>
__global__ void vector_add(T* a, T* b, T* c, uint32_t n) {{
    uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {{
        c[i] = a[i] + b[i];
    }}
}}

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&vector_add<{kwargs["T"]}>);
}}
"""

    @staticmethod
    def launchKernel(kernel: cbd.CUkernel | cbd.CUfunction, **kwargs: Dict):
        """
        Params:
        f
        unsigned int gridDimX, 
        unsigned int gridDimY, 
        unsigned int gridDimZ,
        unsigned int blockDimX, 
        unsigned int blockDimY, 
        unsigned int blockDimZ, 
        unsigned int sharedMemBytes, 
        hStream, 
        kernelParams, 
        void_ptr extra
        """
        assert kwargs["A"].shape == kwargs["B"].shape == kwargs["C"].shape
        assert kwargs["A"].device == kwargs["B"].device == kwargs["C"].device
        assert kwargs["A"].dim() == 1

        gridDimX = (kwargs["A"].numel() + 127) // 128
        gridDimY = 1
        gridDimZ = 1
        blockDimX = 128
        blockDimY = 1
        blockDimZ = 1
        sharedMemBytes = 0
        hStream = kwargs["STREAM"]

        kernelParams = (
            kwargs["A"].data_ptr(),
            kwargs["B"].data_ptr(),
            kwargs["C"].data_ptr(),
            kwargs["A"].numel(),
        )
        # arg_types = (
        #     ctypes.c_void_p,
        #     ctypes.c_void_p,
        #     ctypes.c_void_p,
        #     ctypes.c_uint32,
        # )
        breakpoint()
        kernel_args = (kernel, 
                       gridDimX, gridDimY, gridDimZ,
                       blockDimX, blockDimY, blockDimZ,
                       sharedMemBytes,
                       hStream,
                       kernelParams,
                       [] # extra
                       )
        return CALL_CUDA_FUNC("cuLaunchKernel", *kernel_args) 
    
    # noinspection PyShadowingNames,PyMethodOverriding
    @staticmethod
    def launch(kernel: cbd.CUkernel, **kwargs: Dict[str, Any]) -> cbd.CUresult:
        assert kwargs["A"].shape == kwargs["B"].shape == kwargs["C"].shape
        assert kwargs["A"].device == kwargs["B"].device == kwargs["C"].device
        assert kwargs["A"].dim() == 1

        config = cbd.CUlaunchConfig()
        config.gridDimX = (kwargs["A"].numel() + 127) // 128
        config.gridDimY = 1
        config.gridDimZ = 1
        config.blockDimX = 128
        config.blockDimY = 1
        config.blockDimZ = 1
        config.hStream = kwargs["STREAM"]

        arg_values = (
            kwargs["A"].data_ptr(),
            kwargs["B"].data_ptr(),
            kwargs["C"].data_ptr(),
            kwargs["A"].numel(),
        )
        arg_types = (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
        )

        return cbd.cuLaunchKernelEx(config, kernel, (arg_values, arg_types), 0)[0]


if __name__ == "__main__":
    print("Generated code:")
    kwargs = {"T": "float"}

    if SHOULD_TRACE:
        tracer = create_tracer()
        trace_dir = "traces/jit"
        os.makedirs(trace_dir, exist_ok=True)
        tracer.output_file = os.path.join(trace_dir, "vec_add.generate.json")
    else:
        tracer = nullcontext()
    with tracer:
        code = VectorAddRuntime.generate(kwargs)
    print(code)
    print()

    compiler_name = "NVCC"  # , 'NVRTC'):
    # Get compiler
    compiler_cls = getattr(jit, f"{compiler_name}Compiler")
    print(f"Compiler: {compiler_name}, version: {compiler_cls.__version__()}")

    # Build
    print("Building ...")

    if SHOULD_TRACE:
        tracer.output_file = os.path.join(trace_dir, "vec_add.build.json")
    with tracer:
        func: VectorAddRuntime = compiler_cls.build("test_func", code, VectorAddRuntime, kwargs)
    breakpoint()
    # Run and check
    a = torch.randn((1024,), dtype=torch.float32, device="cuda")
    b = torch.randn((1024,), dtype=torch.float32, device="cuda")
    c = torch.empty_like(a)

    if SHOULD_TRACE:
        tracer.output_file = os.path.join(trace_dir, "vec_add.run.json")
    with tracer:
        ret = func(A=a, B=b, C=c, STREAM=torch.cuda.current_stream().cuda_stream)
    
    assert ret == cbd.CUresult.CUDA_SUCCESS, ret
    torch.testing.assert_close(c, a + b)
    print(f"JIT test for {compiler_name} passed\n")
    
    cufunc = func.create_cuFunc()
    assert cufunc is not None

    c2 = torch.empty_like(a)
    ret = func.launch(cufunc, A=a, B=b, C=c2, STREAM=torch.cuda.current_stream().cuda_stream)
    if ret != CUDA_SUCCESS:
        print(f"Launch not successful: {ret}")
    else:
        torch.testing.assert_close(c, c2)
        print("cuFunc test passed!")