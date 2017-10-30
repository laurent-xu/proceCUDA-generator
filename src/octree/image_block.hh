#pragma once

#include "octree.hh"

class ImageBlock
{
    public:
        ImageBlock();
        virtual ~ImageBlock();
    public:
        int depth;
        int position[3];
        int size[3];
        static const float vector[8][3];
};

void build_image_octree(Octree<ImageBlock>* octMap, const int tree_depth, int vol_dim[3]);
void add_image_blocks(Octree<ImageBlock>* octMap, const int tree_depth);
Octree<ImageBlock>* add_image_block(Octree<ImageBlock>* octMap, int idx);
