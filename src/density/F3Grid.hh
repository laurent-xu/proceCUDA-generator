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

  BOTH_TARGET F3 operator+(const F3& other) const
  {
    return F3{val + other.val, grad + other.grad};
  }

  BOTH_TARGET F3 operator*(const F3& other) const
  {
    return F3{val * other.val, grad * other.val + val * other.grad};
  }
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

  BOTH_TARGET vec3_t to_position(size_t x, size_t y, size_t z) const
  {
    return offset + vec3_t(x, y, z) * precision;
  }
};

template <bool DeviceImplementation>
class GridF3
{
public:
  using dist_t = F3::val_t;
  using vec3_t = F3::vec3_t;
  using grid_t = std::shared_ptr<GridF3<DeviceImplementation>>;

  template <typename... Args>
  static HOST_TARGET grid_t get_grid(Args&&... args)
  {
    return std::shared_ptr<GridF3>(new GridF3(std::forward<Args>(args)...));
  }

  HOST_TARGET ~GridF3()
  {
    if (!hold_)
    {
      if (DeviceImplementation)
        HOST_FREE(points_);
      else
        free(points_);
    }
  }

  BOTH_TARGET size_t dim_size() const { return info_.dimension; }

  DEVICE_TARGET F3& at(size_t x, size_t y, size_t z)
  {
    return points_[x * info_.dimension * info_.dimension +
                   y * info_.dimension + z];
  }

  BOTH_TARGET vec3_t to_position(size_t x, size_t y, size_t z) const
  {
    return info_.to_position(x, y, z);
  }

  BOTH_TARGET F3* get_grid()
  {
    return points_;
  }

  BOTH_TARGET GridInfo get_grid_info()
  {
    return info_;
  }

  HOST_TARGET void hold() { hold_ = true; }
  HOST_TARGET void release() { hold_ = false; }

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
      points_ = (F3*)malloc(size);
  }

  HOST_TARGET GridF3(const GridInfo& info)
    : info_(info)
  {
    auto dimension = info.dimension;
    auto size = dimension * dimension * dimension * sizeof(F3);
    if (DeviceImplementation)
      HOST_MALLOC(points_, size);
    else
      points_ = (F3*)malloc(size);
  }

  F3* points_;
  const GridInfo info_;
  bool hold_ = false;
};

#ifdef CUDA_CODE
static inline GridF3<false>::grid_t
copy_to_host(const GridF3<true>::grid_t& in)
{
  auto result = GridF3<false>::get_grid(in->get_grid_info());
  auto dimension = in->dim_size();
  auto size = dimension * dimension * dimension * sizeof(F3);
  cudaMemcpy((void*)result->get_grid(), (void*)in->get_grid(), size,
             cudaMemcpyDeviceToHost);
  cudaCheckError();
  return result;
}

static inline GridF3<true>::grid_t
copy_to_device(const GridF3<false>::grid_t& in)
{
  auto result = GridF3<true>::get_grid(in->get_grid_info());
  auto dimension = in->dim_size();
  auto size = dimension * dimension * dimension * sizeof(F3);
  cudaMemcpy((void*)result->get_grid(), (void*)in->get_grid(), size,
             cudaMemcpyHostToDevice);
  cudaCheckError();
  return result;
}
#endif
