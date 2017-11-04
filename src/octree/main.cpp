#include <iostream>
#include <cstdlib>
#include "image_block.hh"

void help(char* name)
{
    std::cout << "Usage: " << name << " depth size pos_x pos_y pos_z" << std::endl;
    std::cout << "example:" << std::endl;
    std::cout << name << " 5 100. 60. 60. 60." << std::endl;
    std::cout << "=> Splits an image of 100x100x100 and stores the octree centered in the position (60, 60, 60) with a maximum depth of 5" << std::endl;
}

int main(int argc, char* argv[])
{
    if (argc != 6)
    {
        help(argv[0]);
        exit(0);
    }
    double volume = std::stof(argv[2]);
    double position[3] = {std::stof(argv[3]), std::stof(argv[4]), std::stof(argv[5])};
    Octree<ImageBlock>* octMap = new Octree<ImageBlock>();
    build_positionned_octree(octMap, atoi(argv[1]), volume, position);
    std::cout << "Size: " << octMap->size() << std::endl;
    std::cout << "Depth: " << octMap->depth() << std::endl;
    std::cout << "Leaves: " << octMap->leaves() << std::endl;
    return 0;
}
