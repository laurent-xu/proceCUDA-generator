#pragma once

#include <iostream>
#include <cstdlib>
#include "image_block.hh"
#include "lru.hh"

class Frame
{
    public:
        Frame(double volume, size_t max_depth, size_t cache_memory) : volume(volume), max_depth(max_depth), generated_grids(cache_memory) {};
        OctMap* update(double position[3]);
		size_t get_hash_value(double position[3]);
    private:
        double volume;
        size_t max_depth;
        LRUCache<size_t, OctMap*> generated_grids;
};
