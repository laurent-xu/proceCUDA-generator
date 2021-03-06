#pragma once

#ifdef DEBUG
#define CERR std::cerr
#else
#define CERR while(false) std::cerr
#endif

#ifdef CUDA_CODE

#define cudaCheckError() {                                                     \
  cudaError_t e=cudaGetLastError();                                            \
  if (e != cudaSuccess)                                                        \
  {                                                                            \
    printf("Cuda failure %s:%d: '%s'\n",__FILE__,                              \
           __LINE__,cudaGetErrorString(e));                                    \
    exit(EXIT_FAILURE);                                                        \
  }                                                                            \
}

#define __FORCEINLINE__ __forceinline__
#define BOTH_TARGET __device__ __host__
#define DEVICE_TARGET __device__
#define HOST_TARGET __host__
#define HOST_MALLOC(ptr, size) do {\
    cudaMalloc((void**)&ptr, size);\
    cudaError_t e=cudaGetLastError();                                            \
    if (e != cudaSuccess)                                                        \
    {                                                                            \
      printf("Cuda failure %s:%d: '%s'\n",__FILE__,                              \
             __LINE__,cudaGetErrorString(e));                                    \
      exit(EXIT_FAILURE);                                                        \
    }                                                                            \
  } while(0)
#define HOST_FREE(ptr) do {\
    cudaFree((ptr));       \
    cudaError_t e=cudaGetLastError();                                            \
    if (e != cudaSuccess)                                                        \
    {                                                                            \
      printf("Cuda failure %s:%d: '%s'\n",__FILE__,                              \
             __LINE__,cudaGetErrorString(e));                                    \
      exit(EXIT_FAILURE);                                                        \
    }                                                                            \
  } while(0)
#define CUDA_DEVICE_SYNCRHONIZE() cudaDeviceSynchronize()
#define CUDA_MALLOC_HOST(points, size) do { cudaMallocHost(points, size); } while(false)
#define CUDA_FREE_HOST(points) do { cudaFreeHost(points); } while(false)
#ifdef CUDA_GENERTION
#define GENERATION_TYPE true
#endif
#ifdef CUDA_RENDERING
#define RENDERING_TYPE true
#endif
#else
#define __FORCEINLINE__ inline
#define CUDA_DEVICE_SYNCRHONIZE() do { continue; } while(false)
#define CUDA_MALLOC_HOST(points, size) do { *points = (F3*)malloc(size); } while(false)
#define CUDA_FREE_HOST(points) do { free(points); } while(false)
#define BOTH_TARGET
#define DEVICE_TARGET
#define HOST_TARGET
#define HOST_MALLOC(ptr, size) do {      \
    ptr = (decltype (ptr))malloc((size));\
  } while(0)
#define HOST_FREE(ptr) do {              \
    free(ptr);                           \
  } while(0)
#define GENRATION_TYPE false
#define RENDERING_TYPE false
#endif
