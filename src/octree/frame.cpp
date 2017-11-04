#include "frame.hh"

size_t Frame::get_hash_value(double position[3])
{
	size_t h1 = std::hash<double>()(position[0]);
    size_t h2 = std::hash<double>()(position[1]);
    size_t h3 = std::hash<double>()(position[2]);
    return (h1 ^ (h2 << 1)) ^ h3;
}

OctMap* Frame::update(double position[3])
{
	size_t hash_value = get_hash_value(position);
    if (!this->generated_grids.exist(hash_value))
    {
        OctMap* new_octMap = new OctMap();
        build_positionned_octree(new_octMap, this->max_depth, this->volume, position);
        this->generated_grids.put(hash_value, new_octMap);
        return new_octMap;
    }
    return this->generated_grids.get(hash_value);
}
