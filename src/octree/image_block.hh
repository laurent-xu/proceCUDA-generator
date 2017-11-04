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
        double distance(double position[3]);
        bool contains(double position[3]);
        int depth;
        double position[3];
        double size;
        bool generated;
        static const double vector[8][3];
};

void build_image_octree(Octree<ImageBlock>* octMap, const int tree_depth, double volume);
void build_positionned_octree(Octree<ImageBlock>* octMap, const int tree_depth, double volume, double position[3]);
void add_image_blocks(Octree<ImageBlock>* octMap, const int tree_depth);
Octree<ImageBlock>* add_image_block(Octree<ImageBlock>* octMap, int idx);
void get_near_blocks(std::vector<ImageBlock*>& res, Octree<ImageBlock>* octMap, double position[3], double threshold);
