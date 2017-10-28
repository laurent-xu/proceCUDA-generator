#pragma once
#define GRIDSIZE 32
#include <glm/glm.hpp>
#include <cassert>
#include <memory>
#include <vector>

namespace density {}

struct F3
{
  using val_t = double;
  using dist_t = double;
  using vec3_t = glm::tvec3<dist_t>;
  val_t val;
  vec3_t grad;
};

class GridF3;

class GridF3Data
{
public:
  using data_t = std::shared_ptr<GridF3Data>;
  using const_data_t = std::shared_ptr<const GridF3Data>;

  static data_t get_grid()
  {
    if (!free_.empty())
    {
      auto result = free_.back();
      free_.pop_back();
      return result;
    }
    return std::make_shared<GridF3Data>();
  }

  static void release(const data_t& data)
  {
    free_.push_back(data);
  }

private:
  friend class GridF3;
  static std::vector<data_t> free_;
  F3 points_[GRIDSIZE][GRIDSIZE][GRIDSIZE];
};

class GridF3
{
public:
  using dist_t = F3::val_t;
  using vec3_t = F3::vec3_t;
  using data_t = GridF3Data::data_t;
  using const_data_t = GridF3Data::const_data_t;

  GridF3(const dist_t& precision, const vec3_t& offset)
    : data_(GridF3Data::get_grid()),
      precision_(precision),
      offset_(offset) {}

  ~GridF3()
  {
    GridF3Data::release(data_);
  }

  size_t dim_size() const { return GRIDSIZE; }

  F3& at(size_t x, size_t y, size_t z)
  {
    return data_->points_[x][y][z];
  }

  F3 at(size_t x, size_t y, size_t z) const
  {
    return data_->points_[x][y][z];
  }

  vec3_t to_position(size_t x, size_t y, size_t z) const
  {
    return offset_ + vec3_t(x, y, z) * precision_;
  }

  data_t get_grid()
  {
    return data_;
  }

  const_data_t get_grid() const
  {
    return std::const_pointer_cast<const GridF3Data>(data_);
  }

private:
  GridF3Data::data_t data_;
  dist_t precision_;
  vec3_t offset_;
};
