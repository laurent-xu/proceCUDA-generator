#include <iostream>
#include <cstdlib>
#include "image_block.hh"
#include "frame.hh"

void help(char* name)
{
    std::cout << "Usage: " << name << " depth size pos_x pos_y pos_z" << std::endl;
    std::cout << "example:" << std::endl;
    std::cout << name << " 5 100. 60. 60. 60." << std::endl;
    std::cout << "=> Splits an image of 100x100x100 and stores the octree centered in the position (60, 60, 60) with a maximum depth of 5" << std::endl;
}

void info_frame(Frame& frame, double position[3])
{
    OctMap* octMap = frame.update(position);
    std::cout << "Size: " << octMap->size() << std::endl;
    std::cout << "Depth: " << octMap->depth() << std::endl;
    std::cout << "Leaves: " << octMap->leaves() << std::endl;
}

int main(int argc, char* argv[])
{
    if (argc != 6)
    {
        help(argv[0]);
        exit(0);
    }
	int depth = atoi(argv[1]);
    double volume = std::stof(argv[2]);
    double position[3] = {std::stof(argv[3]), std::stof(argv[4]), std::stof(argv[5])};
	size_t cache_memory = 10;
    Frame frame(volume, depth, cache_memory);
	info_frame(frame, position);
	std::cout << "Inserting same position" << std::endl;
	info_frame(frame, position);
	position[0]-= 40;
	std::cout << "Inserting new position" << std::endl;
	info_frame(frame, position);
    return 0;
}
