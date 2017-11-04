#pragma once
#include <utils/cudamacro.hh>
#include <glm/glm.hpp>
#include <cassert>
#include <memory>
#include <vector>
#include <cstdlib>

struct F3
{
  using val_t = double;
  using dist_t = double;
  using vec3_t = glm::tvec3<dist_t>;
  val_t val;
  vec3_t grad;
};

struct GridInfo
{
  using dist_t = double;
  using vec3_t = glm::tvec3<dist_t>;
  GridInfo(dist_t precision, vec3_t offset, size_t dimension)
    : precision(precision),
      offset(offset),
      dimension(dimension)
  {
  }

  dist_t precision;
  vec3_t offset;
  size_t dimension;
};

template <bool DeviceImplementation>
class GridF3
{
public:
  using dist_t = F3::val_t;
  using vec3_t = F3::vec3_t;
  using grid_t = std::shared_ptr<GridF3>;

#ifdef CUDA_CODE
  __device__ GridF3(const dist_t& precision,
                    const vec3_t& offset,
                    typename std::enable_if<DeviceImplementation, size_t>::type
                    dimension)
    : info_(precision, offset, dimension)
  {
    auto size = dimension * dimension * dimension * sizeof(F3);
    points_ = malloc(size);
  }

  __device__ GridF3(typename std::enable_if<DeviceImplementation,
                                            const GridInfo&>::type info)
    : info_(info)
  {
    auto dimension = info.dimension;
    auto size = dimension * dimension * dimension * sizeof(F3) * ;
    points_ = malloc(size);
  }

  __device__ ~GridF3()
    : info_(info)
  {
    free(points_);
  }

  __host__ grid_t<false> copy_to_host()
  {
    auto result = GridF3<false>::get_grid(info_);
    auto dimension = info_.dimension;
    auto size = dimension * dimension * dimension * sizeof(F3);
    if (DeviceImplementation)
      cudaMemcpy((void*)result->points_,
                 (void*)points_, size,
                 cudaMemcpyDeviceToHost);
    else
    {
      std::cerr << "Unecessary host to host copy" << std::endl;
      memcpy(result->points_, points_, size);
    }
  }

  __host__ grid_t<true> copy_to_device()
  {
    auto result = GridF3<true>::get_grid(info_);
    auto dimension = info_.dimension;
    auto size = dimension * dimension * dimension * sizeof(F3);
    if (DeviceImplementation)
      cudaMemcpy((void*)result->points_,
                 (void*)points_, size,
                 cudaMemcpyHostToDevice);
    else
    {
      std::cerr << "Unecessary device to device copy" << std::endl;
      memcpy(result->points_, points_, size);
    }
  }
#endif

  template <typename... Args>
  static HOST_TARGET grid_t get_grid(Args&&... args)
  {
    return std::shared_ptr<GridF3>(new GridF3(std::forward<Args>(args)...));
  }

  HOST_TARGET ~GridF3()
  {
    if (DeviceImplementation)
      HOST_FREE(points_);
    else
      delete[] points_;
  }

  BOTH_TARGET size_t dim_size() const { return info_.dimension; }

  DEVICE_TARGET F3& at(size_t x, size_t y, size_t z)
  {
    return points_[x * info_.dimension * info_.dimension +
                   y * info_.dimension + z];
  }

  BOTH_TARGET vec3_t to_position(size_t x, size_t y, size_t z) const
  {
    return info_.offset + vec3_t(x, y, z) * info_.precision;
  }

  BOTH_TARGET F3* get_grid()
  {
    return points_;
  }

  BOTH_TARGET GridInfo get_grid_info()
  {
    return info_;
  }

private:
  friend class GridF3<true>;
  friend class GridF3<false>;

  HOST_TARGET GridF3(const dist_t& precision,
                     const vec3_t& offset,
                     size_t dimension)
    : info_(precision, offset, dimension)
  {
    auto size = dimension * dimension * dimension * sizeof(F3);
    if (DeviceImplementation)
      HOST_MALLOC(points_, size);
    else
      points_ = new F3[size];
  }

  HOST_TARGET GridF3(const GridInfo& info)
    : info_(info)
  {
    auto dimension = info.dimension;
    auto size = dimension * dimension * dimension * sizeof(F3);
    if (DeviceImplementation)
      HOST_MALLOC(points_, size);
    else
      points_ = new F3[size];
  }

  F3* points_;
  const GridInfo info_;
};
