#include "image_block.hh"

ImageBlock::ImageBlock()
{
    this->depth = 0;
    for (int i = 0; i < 3; ++i)
    {
        this->position[i] = 0;
        this->size[i] = 0;
    }
}

ImageBlock::~ImageBlock() {}

const float ImageBlock::vector[8][3] =
{
    {0, 0, 0},
    {0.5, 0, 0},
    {0, 0.5, 0},
    {0, 0, 0.5},
    {0.5, 0.5, 0},
    {0, 0.5, 0.5},
    {0.5, 0, 0.5},
    {0.5, 0.5, 0.5},
};

void build_image_octree(Octree<ImageBlock>* octMap, const int tree_depth, int vol_dim[3])
{
    auto node = new ImageBlock();
    octMap->set_node(node);
    for (int i = 0; i < 3; i++)
        node->size[i] = vol_dim[i];
    add_image_blocks(octMap, tree_depth);
}

void add_image_blocks(Octree<ImageBlock>* octMap, const int depth)
{
    if (octMap->get_node()->depth < depth)
        for (int i = 0; i < 8; i++)
            add_image_blocks(add_image_block(octMap, i), depth);
}

Octree<ImageBlock>* add_image_block(Octree<ImageBlock>* octMap, int idx)
{
    auto parent_node = octMap->get_node();
    octMap->add_child(idx, new Octree<ImageBlock>());
    auto node = new ImageBlock();
    node->depth = parent_node->depth + 1;
    for (int i = 0; i < 3; i++)
    {
        node->position[i] = (int)(parent_node->position[i] + ImageBlock::vector[idx][i] * parent_node->size[i]);
        node->size[i] = (int)(parent_node->size[i] / 2);
    }
    octMap->get_child(idx)->set_node(node);
    return octMap->get_child(idx);
}
