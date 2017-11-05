#pragma once
#include <density/F3Grid.hh>
#include <vector>

#ifdef CUDA_GENERATION
__global__ void kernel_f3_caller(GridF3<true> grid);
#else
F3 kernel_f3(const F3::vec3_t& position);
#endif
