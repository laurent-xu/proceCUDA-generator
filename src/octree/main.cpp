#include <iostream>
#include <cstdlib>
#include "image_block.hh"

void help(char* name)
{
    std::cout << "Usage: " << name << " depth size_x size_y size_z threshold pos_x pos_y pos_z" << std::endl;
    std::cout << "example:" << std::endl;
    std::cout << name << " 3 100 100 100 30. 110 110 110" << std::endl;
    std::cout << "=> Splits an image of 100x100x100 into 3^8 = 512 blocks and stores the blocks near with a distance < 30. of the position (110, 110, 110)" << std::endl;
}

int main(int argc, char* argv[])
{
    if (argc != 9)
    {
        help(argv[0]);
        exit(0);
    }
    Octree<ImageBlock>* octMap = new Octree<ImageBlock>();
    int vol_dim[3] = {atoi(argv[2]), atoi(argv[3]), atoi(argv[4])};
    build_image_octree(octMap, atoi(argv[1]), vol_dim);
    std::cout << "Size: " << octMap->size() << std::endl;
    std::cout << "Depth: " << octMap->depth() << std::endl;
    std::cout << "Leaves: " << octMap->leaves() << std::endl;
    std::vector<ImageBlock*> near_blocks;
    float threshold = std::stof(argv[5]);
    float position[3] = {std::stof(argv[6]), std::stof(argv[7]), std::stof(argv[8])};
    get_near_blocks(near_blocks, octMap, position, threshold);
    for (int i = 0; i < near_blocks.size(); ++i)
    {
        for (int j = 0; j < 3; ++j)
            std::cout << near_blocks[i]->position[j] << " ";
        std::cout << std::endl;
        for (int j = 0; j < 3; ++j)
            std::cout << near_blocks[i]->size[j] << " ";
        std::cout << std::endl;
        std::cout << std::endl;
    }
    return 0;
}
