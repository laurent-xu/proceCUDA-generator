#pragma once

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

#define BOTH_TARGET __device__ __host__
#define DEVICE_TARGET __device__
#define HOST_TARGET __host__
#define HOST_MALLOC(ptr, size) do {\
    cudaMalloc((void**)&ptr, size);\
  } while(0)
#define HOST_FREE(ptr) do {\
    cudaFree((ptr));       \
  } while(0)
#else
#define BOTH_TARGET   /**/
#define DEVICE_TARGET /**/
#define HOST_TARGET   /**/
#define HOST_MALLOC(ptr, size) do {      \
    ptr = (decltype (ptr))malloc((size));\
  } while(0)
#define HOST_FREE(ptr) do {              \
    free(ptr);                           \
  } while(0)
#endif
