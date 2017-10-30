#include <algorithm>

template<typename Node>
Octree<Node>::Octree()
{
    this->node = nullptr;
    for (int i = 0; i < 8; i++)
        this->children[i] = nullptr;
}

template<typename Node>
Octree<Node>::~Octree()
{
    for (int i = 0; i < 8; i++)
    {
        delete this->children[i];
        this->children[i] = nullptr;
    }
}

template<typename Node>
size_t Octree<Node>::depth()
{
    size_t depth = 0;
    for (int i = 0; i < 8; ++i)
        if (this->children[i])
            depth = std::max(depth, 1 + this->children[i]->depth());
    return depth;
}

template<typename Node>
size_t Octree<Node>::size()
{
    size_t size = 0;
    for (int i = 0; i < 8; ++i)
        if (this->children[i])
            size += 1 + this->children[i]->size();
    return size;
}

template<typename Node>
size_t Octree<Node>::leaves()
{
    size_t size = 0;
    if (this->leaf())
        return 1;
    for (int i = 0; i < 8; ++i)
        if (this->children[i])
            size += this->children[i]->leaves();
    return size;
}

template<typename Node>
bool Octree<Node>::leaf()
{
    for (int i = 0; i < 8; ++i)
        if (children[i])
            return false;
    return true;
}
