#pragma once

#include "octree.hh"
#include <cmath>
#include <vector>

class ImageBlock
{
    public:
        ImageBlock();
        virtual ~ImageBlock();
    public:
        float distance(float position[3]);
        int depth;
        float position[3];
        int size[3];
        bool generated;
        static const float vector[8][3];
};

void build_image_octree(Octree<ImageBlock>* octMap, const int tree_depth, int vol_dim[3]);
void add_image_blocks(Octree<ImageBlock>* octMap, const int tree_depth);
Octree<ImageBlock>* add_image_block(Octree<ImageBlock>* octMap, int idx);
void get_near_blocks(std::vector<ImageBlock*>& res, Octree<ImageBlock>* octMap, float position[3], float threshold);
