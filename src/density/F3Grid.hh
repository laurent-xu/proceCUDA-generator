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

class GridInfo
{
  public:
  using dist_t = double;
  using vec3_t = glm::tvec3<int>;
  GridInfo(dist_t precision, vec3_t offset, size_t dimension)
    : precision(precision),
      offset(offset),
      dimension(dimension)
  {
  }

  GridInfo()
    : precision(0),
      offset({0, 0, 0}),
      dimension(0)
  {
  }

  public:
  dist_t precision;
  vec3_t offset;
  size_t dimension;

  public:
  BOTH_TARGET F3::vec3_t to_position(size_t x, size_t y, size_t z) const
  {
    return F3::vec3_t(offset * int(dimension) + vec3_t(x, y, z)) * precision;
  }

  bool operator ==(const GridInfo &b) const
  {
    return this->dimension == b.dimension
    && this->offset == b.offset
    && this->precision == b.precision;
  }
};

template <bool DeviceImplementation>
class GridF3
{
public:
  using dist_t = F3::val_t;
  using off_vec3_t = GridInfo::vec3_t;
  using vec3_t = F3::vec3_t;
  using grid_t = std::shared_ptr<GridF3<DeviceImplementation>>;

  template <typename... Args>
  static HOST_TARGET grid_t get_grid(Args&&... args)
  {
    return std::shared_ptr<GridF3>(new GridF3(std::forward<Args>(args)...));
  }

  HOST_TARGET ~GridF3()
  {
    if (DeviceImplementation)
    {
      if (!hold_)
        free_grid(points_);
    }
    else
      free_grid(points_);
  }

  BOTH_TARGET size_t dim_size() const { return info_.dimension; }

  BOTH_TARGET F3& at(size_t x, size_t y, size_t z)
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

  static std::vector<F3*>& get_memory_pool()
  {
    static std::vector<F3*> memory_pool;
    return memory_pool;

  }

  static F3* alloc_grid(size_t size)
  {
    bool first = true;
    static size_t old_size;
    if (first)
    {
      old_size = size;
      first = false;
    }
    if (old_size != size)
      std::abort();
    F3* result;
    if (get_memory_pool().empty())
    {
      if (DeviceImplementation)
        HOST_MALLOC(result, size);
      else
        CUDA_MALLOC_HOST(&result, size);
    }
    else
    {
      result = get_memory_pool().back();
      get_memory_pool().pop_back();
    }
    return result;
  }

  static void free_grid(F3* to_be_freed)
  {
    get_memory_pool().push_back(to_be_freed);
  }

  HOST_TARGET GridF3(const dist_t& precision,
                     const off_vec3_t& offset,
                     size_t dimension)
    : info_(precision, offset, dimension)
  {
    auto size = dimension * dimension * dimension * sizeof(F3);
    points_ = alloc_grid(size);
  }

  HOST_TARGET GridF3(const GridInfo& info)
    : info_(info)
  {
    auto dimension = info.dimension;
    auto size = dimension * dimension * dimension * sizeof(F3);
    points_ = alloc_grid(size);
  }

  F3* points_;
  const GridInfo info_;
  bool hold_ = false;
};

#ifdef CUDA_CODE
static inline GridF3<false>::grid_t
copy_to_host_async(const GridF3<true>::grid_t& in_d, cudaStream_t& stream)
{
  auto result_h = GridF3<false>::get_grid(in_d->get_grid_info());
  auto dimension = in_d->dim_size();
  auto size = dimension * dimension * dimension * sizeof(F3);
  cudaMemcpyAsync((void*)result_h->get_grid(), (void*)in_d->get_grid(), size,
                   cudaMemcpyDeviceToHost, stream);
  return result_h;
}

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
