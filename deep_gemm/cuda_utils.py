import cuda.bindings.driver as cbd

CUDA_SUCCESS = cbd.CUresult.CUDA_SUCCESS
GET_MODULE = "cuLibraryGetModule"
MODULE_FUNCTIONS = "cuModuleEnumerateFunctions"
MODULE_FUNCTION_COUNT = "cuModuleGetFunctionCount"
LIBRARY_KERNELS = "cuLibraryEnumerateKernels"
LIBRARY_KERNEL_COUNT = "cuLibraryGetKernelCount"

def CALL_CUDA_FUNC(func_name, *args, **kwargs):
    try:
        fn = getattr(cbd, func_name, None)
        if fn is None:
            print(f"Could not fund {func_name} in cuda bindings")
            return None

        result = fn(*args, **kwargs)
        if len(result) > 1:
            result, rest = result
        else:
            rest = None
        
        if not result == CUDA_SUCCESS:
            print(f"{func_name} returned cuda error: {result}")
            return None
        return rest or result
    except Exception as e:
        print(f"Error while calling {func_name}: {e}")
        return None
