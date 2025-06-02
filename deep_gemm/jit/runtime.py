import os
import subprocess
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional, Type

import cuda.bindings.driver as cbd
import torch
from torch.utils.cpp_extension import CUDA_HOME

from deep_gemm.cuda_utils import (
    CALL_CUDA_FUNC,
    LIBRARY_KERNEL_COUNT,
    LIBRARY_KERNELS,
    MODULE_FUNCTION_COUNT,
    MODULE_FUNCTIONS,
)


class Runtime:
    def __init__(self, path: str) -> None:
        self.path = path
        self.lib = None
        self.kernel = None
        self.func = None
        assert self.is_path_valid(self.path)

    @staticmethod
    def is_path_valid(path: str) -> bool:
        # Exists and is a directory
        if not os.path.exists(path) or not os.path.isdir(path):
            return False

        # Contains all necessary files
        files = ["kernel.cubin"]
        return all(os.path.exists(os.path.join(path, file)) for file in files)

    @staticmethod
    def generate(kwargs: Dict[str, Any]) -> str:
        raise NotImplemented

    @staticmethod
    def launch(kernel: cbd.CUkernel, kwargs: Dict[str, Any]) -> cbd.CUresult:
        raise NotImplemented

    def create_cuFunc(self, **kwargs):
        if self.func is None:
            path = bytes(os.path.join(self.path, "kernel.cubin"), "utf-8")
    
            # https://nvidia.github.io/cuda-python/cuda-bindings/latest/module/driver.html#cuda.bindings.driver.cuLibraryLoadFromFile
            print(f"Loading cuLibrary from {path}")
            result, self.lib = cbd.cuLibraryLoadFromFile(
                path,
                [],  # jitOptions
                [],  # jitOptionsValues
                0,  # numJitOptions
                [],  # libraryOptions
                [],  # libraryOptionValues
                0,  # numLibraryOptions
            )
            assert result == cbd.CUresult.CUDA_SUCCESS, (
                f"Failed to load library: {result}"
            )

            # Extract the kernel name
            # TODO: use `cuda-bindings` API to do this (requires at least 12.8)
            command = [f"{CUDA_HOME}/bin/cuobjdump", "-symbols", path]
            result = subprocess.run(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            assert result.returncode == 0
            illegal_names = [
                "vprintf",
                "__instantiate_kernel",
                "__internal",
                "__assertfail",
            ]
            check_illegal = lambda line: any([name in line for name in illegal_names])
            print(f"cuobjdump symbols:\n{result.stdout}")

            kernel_names = [
                line.split()[-1]
                for line in result.stdout.splitlines()
                if line.startswith("STT_FUNC") and not check_illegal(line)
            ]
            
            print(f"Parsed kernel names: {'\n\t'.join(kernel_names)}")
            assert len(kernel_names) == 1, (
                f"Too many kernels in the library: {kernel_names}"
            )
            
            
            # https://nvidia.github.io/cuda-python/cuda-bindings/latest/module/driver.html#cuda.bindings.driver.cuLibraryEnumerateKernels
            # https://nvidia.github.io/cuda-python/cuda-bindings/latest/module/driver.html#cuda.bindings.driver.cuLibraryGetKernelCount
            # Load kernel from the library

            # num_kernels = CALL_CUDA_FUNC(LIBRARY_KERNEL_COUNT, self.lib)      
            # #assert num_kernels == 1, (f"Found {num_kernels} kernels!")
            
            mod = CALL_CUDA_FUNC("cuLibraryGetModule", self.lib)
            
            # num_kernels = 1
            # kernel_handles = CALL_CUDA_FUNC(LIBRARY_KERNELS, num_kernels, self.lib)
            
            # if kernel_handles is not None:
            #     kernel_handle = kernel_handles[0]
            #     self.kernel_handle = kernel_handle
            
            num_kernels = 1
            # func_count_lib = CALL_CUDA_FUNC(MODULE_FUNCTION_COUNT, self.lib)
            func_count = CALL_CUDA_FUNC(MODULE_FUNCTION_COUNT, mod)
            
            assert func_count == 1, f"Found {func_count} functions!"
            
            # func_names_lib = CALL_CUDA_FUNC(MODULE_FUNCTIONS, num_kernels, self.lib)
            func_handles = CALL_CUDA_FUNC(MODULE_FUNCTIONS, num_kernels, mod)
            
            if func_handles is not None:
                func_handle = func_handles[0]
                self.func = func_handle

        return self.func
    
    def __call__(self, **kwargs) -> cbd.CUresult:
        """
        Params for `cuLibraryLoadFromFile`:
        fileName (bytes) - File to load from
        jitOptions (List[CUjit_option]) - Options for JIT
        jitOptionsValues (List[Any]) - Option values for JIT
        numJitOptions (unsigned int) - Number of options
        libraryOptions (List[CUlibraryOption]) - Options for loading
        libraryOptionValues (List[Any]) - Option values for loading
        numLibraryOptions (unsigned int) - Number of options for loading
        """
        # Load CUBIN
        if self.kernel is None:
            start_time = time.time_ns()

            # Load CUBIN
            path = bytes(os.path.join(self.path, "kernel.cubin"), "utf-8")
            # https://nvidia.github.io/cuda-python/cuda-bindings/latest/module/driver.html#cuda.bindings.driver.cuLibraryLoadFromFile

            result, self.lib = cbd.cuLibraryLoadFromFile(
                path,
                [],  # jitOptions
                [],  # jitOptionsValues
                0,  # numJitOptions
                [],  # libraryOptions
                [],  # libraryOptionValues
                0,  # numLibraryOptions
            )
            assert result == cbd.CUresult.CUDA_SUCCESS, (
                f"Failed to load library: {result}"
            )

            # Extract the kernel name
            # TODO: use `cuda-bindings` API to do this (requires at least 12.8)
            command = [f"{CUDA_HOME}/bin/cuobjdump", "-symbols", path]
            result = subprocess.run(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            assert result.returncode == 0
            illegal_names = [
                "vprintf",
                "__instantiate_kernel",
                "__internal",
                "__assertfail",
            ]
            check_illegal = lambda line: any([name in line for name in illegal_names])
            kernel_names = [
                line.split()[-1]
                for line in result.stdout.splitlines()
                if line.startswith("STT_FUNC") and not check_illegal(line)
            ]
            assert len(kernel_names) == 1, (
                f"Too many kernels in the library: {kernel_names}"
            )
                      
            print(f"{__file__}: Loading kernel name {kernel_names[0]}")
            result, self.kernel = cbd.cuLibraryGetKernel(
                self.lib, bytes(kernel_names[0], encoding="utf-8")
            )
            assert result == cbd.CUresult.CUDA_SUCCESS, (
                f"Failed to load kernel: {result}"
            )

            end_time = time.time_ns()
            elapsed_time = (end_time - start_time) / 1e6
            if int(os.getenv("DG_JIT_DEBUG", 0)):
                print(f"Loading JIT runtime {self.path} took {elapsed_time:.2f} ms.")

        # noinspection PyArgumentList
        return self.launch(self.kernel, **kwargs)

    def __del__(self) -> None:
        if self.lib is not None:
            res = cbd.cuLibraryUnload(self.lib)[0]
            if res != cbd.CUresult.CUDA_SUCCESS:
                raise Exception(f"Failed to unload library {self.path}: {res}")


class RuntimeCache:
    def __init__(self) -> None:
        self.cache = {}

    def __setitem__(self, path: str, runtime: Runtime) -> None:
        self.cache[path] = runtime

    def get(
        self,
        path: str,
        runtime_cls: Type[Runtime],
        name: str = "",
        kwargs: Dict[str, Any] = None,
        force_enable_cache: bool = False,
    ) -> Optional[Runtime]:
        # In Python runtime
        if path in self.cache:
            print(f"{__file__}: Loading runtime from self.cache at {path}")
            return self.cache[path]

        # Already compiled
        use_cache = force_enable_cache or not int(os.getenv("DG_JIT_DISABLE_CACHE", 0))
        print(f"{__file__}: {use_cache=}")
        
        if use_cache and os.path.exists(path) and Runtime.is_path_valid(path):
            # Print heuristic for the first time
            if name and (
                int(os.getenv("DG_JIT_DEBUG", 0))
                or int(os.getenv("DG_PRINT_CONFIGS", 0))
            ):
                simplified_kwargs = dict()
                for key, value in (
                    kwargs.items() if kwargs is not None else dict().items()
                ):
                    value = (
                        f"torch.Tensor<{value.dtype}>"
                        if isinstance(value, torch.Tensor)
                        else value
                    )
                    value = (
                        f"cuda.bindings.driver.CUtensorMap"
                        if isinstance(value, cbd.CUtensorMap)
                        else value
                    )
                    simplified_kwargs[key] = value
                print(f"Put kernel {name} with {simplified_kwargs} into runtime cache")

            print(f"{__file__}: Creating {runtime_cls.__name__} from {path}")
            runtime = runtime_cls(path)
            print(f"{__file__}: Setting {path} in self.cache")
            self.cache[path] = runtime
            return runtime
        return None
