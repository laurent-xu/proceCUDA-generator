#include "image_block.hh"

ImageBlock::ImageBlock()
{
    this->depth = 0;
    this->size = 0.;
    for (int i = 0; i < 3; ++i)
        this->position[i] = 0.;
    this->generated = false;
}

ImageBlock::~ImageBlock() {}

const double ImageBlock::vector[8][3] =
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

void build_positionned_octree(Octree<ImageBlock>* octMap, const int tree_depth, double volume, double position[3])
{
    auto node = new ImageBlock();
    octMap->set_node(node);
    node->size = volume;
    auto current_tree = octMap;
    for (int i = 0; i < tree_depth; ++i)
    {
        add_image_blocks(current_tree, i + 1);
        for (int j = 0; j < 8; ++j)
        {
            if (current_tree->get_child(j)->get_node()->contains(position))
            {
                current_tree = current_tree->get_child(j);
                break;
            }
        }
    }
}

bool ImageBlock::contains(double position[3])
{
    return position[0] >= this->position[0] && position[0] < this->position[0] + this->size
    && position[1] >= this->position[1] && position[1] < this->position[1] + this->size
    && position[2] >= this->position[2] && position[2] < this->position[2] + this->size;
}

void build_image_octree(Octree<ImageBlock>* octMap, const int tree_depth, double volume)
{
    auto node = new ImageBlock();
    octMap->set_node(node);
    node->size = volume;
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
    node->size = (int)(parent_node->size / 2);
    for (int i = 0; i < 3; i++)
    {
        node->position[i] = (parent_node->position[i] + ImageBlock::vector[idx][i] * parent_node->size);
    }
    octMap->get_child(idx)->set_node(node);
    return octMap->get_child(idx);
}

double ImageBlock::distance(double position[3])
{
    double median[3] = {0., 0., 0.};
    double distance = 0.;
    for (int i = 0; i < 3; ++i)
    {
        median[i] = this->position[i] + this->size / 2;
        distance += (median[i] - position[i]) * (median[i] - position[i]);
    }
    return sqrt(distance);
}

void get_near_blocks(std::vector<ImageBlock*>& res, Octree<ImageBlock>* octMap, double position[3], double threshold)
{
    auto node = octMap->get_node();
    if (octMap->leaf() && node->distance(position) < threshold)
    {
        res.push_back(node);
        return;
    }
    if (node->distance(position) < threshold * pow(2, octMap->depth()))
        for (int i = 0; i < 8; i++)
            get_near_blocks(res, octMap->get_child(i), position, threshold);
}
