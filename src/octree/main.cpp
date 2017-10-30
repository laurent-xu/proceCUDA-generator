#include <iostream>
#include <cstdlib>
#include "image_block.hh"

void help(char* name)
{
    std::cout << "Usage: " << name << " depth size_x size_y size_z" << std::endl;
    std::cout << "example:" << std::endl;
    std::cout << name << " 3 100 100 100" << std::endl;
    std::cout << "=> Splits an image of 100x100x100 into 3^8 = 512 blocks" << std::endl;
}

int main(int argc, char* argv[])
{
    if (argc != 5)
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
    std::cout << "First floor: " << std::endl;
    for (int i = 0; i < 8; ++i)
    {
        for (int j = 0; j < 3; ++j)
            std::cout << octMap->get_child(i)->get_node()->position[j] << " ";
        std::cout << std::endl;
        for (int j = 0; j < 3; ++j)
            std::cout << octMap->get_child(i)->get_node()->size[j] << " ";
        std::cout << std::endl;
        std::cout << std::endl;
    }
    return 0;
}
