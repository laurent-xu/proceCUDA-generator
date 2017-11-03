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

class GridF3
{
public:
  using dist_t = F3::val_t;
  using vec3_t = F3::vec3_t;
  using grid_t = std::shared_ptr<GridF3>;

#ifdef CUDA_GENERATION
  __device__ GridF3(const dist_t& precision,
                    const vec3_t& offset,
                    size_t dimension)
    : info_(precision, offset, dimension)
  {
    points_ = malloc(sizeof(F3) * dimension * dimension * dimension);
  }

  __device__ GridF3(const GridInfo& info)
    : info_(info)
  {
    auto dimension = info.dimension;
    points_ = malloc(sizeof(F3) * dimension * dimension * dimension);
  }

  __device__ GridF3(const GridInfo& info)
    : info_(info)
  {
    free(points_);
  }
#endif

  template <typename... Args>
  static HOST_TARGET_GENERATION grid_t get_grid(Args&&... args)
  {
    return std::make_shared<GridF3>(new GridF3(std::forward(args)...));
  }

  HOST_TARGET_GENERATION ~GridF3()
  {
    HOST_FREE_GENERATION(points_);
  }

  BOTH_TARGET_GENERATION size_t dim_size() const { return info_.dimension; }

  DEVICE_TARGET_GENERATION F3& at(size_t x, size_t y, size_t z)
  {
    return points_[x * info_.dimension * info_.dimension +
                   y * info_.dimension + z];
  }

  BOTH_TARGET_GENERATION vec3_t to_position(size_t x, size_t y, size_t z) const
  {
    return info_.offset + vec3_t(x, y, z) * info_.precision;
  }

  BOTH_TARGET_GENERATION F3* get_grid()
  {
    return points_;
  }

  BOTH_TARGET_GENERATION GridInfo get_grid_info()
  {
    return info_;
  }

private:
  HOST_TARGET_GENERATION GridF3(const dist_t& precision,
                                const vec3_t& offset,
                                size_t dimension)
    : info_(precision, offset, dimension)
  {
    HOST_MALLOC_GENERATION(points_,
                           sizeof(F3) * dimension * dimension * dimension);
  }

  HOST_TARGET_GENERATION GridF3(const GridInfo& info)
    : info_(info)
  {
    auto dimension = info.dimension;
    HOST_MALLOC_GENERATION(points_,
                           sizeof(F3) * dimension * dimension * dimension);
  }

  F3* points_;
  const GridInfo info_;
};
