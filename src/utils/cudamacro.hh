#pragma once

#ifdef CUDA_GENERATION
#define BOTH_TARGET_GENERATION __device__ __host__
#define DEVICE_TARGET_GENERATION __device__
#define HOST_TARGET_GENERATION __host__
#define HOST_MALLOC_GENERATION(ptr, size) do {\
    cudaMalloc((void**)&ptr, size);           \
  } while(0)
#define HOST_FREE_GENERATION(ptr) do {\
    cudaFree((ptr));                  \
  } while(0)
#else
#define BOTH_TARGET_GENERATION /**/
#define DEVICE_TARGET_GENERATION /**/
#define HOST_TARGET_GENERATION /**/
#define HOST_MALLOC_GENERATION(ptr, size) do {\
    ptr = (decltype (ptr))malloc((size));     \
  } while(0)
#define HOST_FREE_GENERATION(ptr) do {\
    free(ptr);                        \
  } while(0)
#endif
