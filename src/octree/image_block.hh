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

typedef Octree<ImageBlock> OctMap;

void build_image_octree(OctMap* octMap, const int tree_depth, double volume);
void build_positionned_octree(OctMap* octMap, const int tree_depth, double volume, double position[3]);
void add_image_blocks(OctMap* octMap, const int tree_depth);
OctMap* add_image_block(OctMap* octMap, int idx);
void get_near_blocks(std::vector<ImageBlock*>& res, OctMap* octMap, double position[3], double threshold);
