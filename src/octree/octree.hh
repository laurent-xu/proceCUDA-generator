#include "math.h"
#include <cstring>

template<typename Node>
class Octree
{
    public:
        Octree();
        virtual ~Octree();
    public:
        Node* node;
        Octree* children[8];
    public:
        void add_child(int idx, Octree* child) {this->children[idx] = child;}
        Octree* get_child(int idx) {return children[idx];}
        void set_node(Node* node) {this->node = node;}
        Node* get_node() {return this->node;}
        size_t depth();
        size_t size();
        size_t leaves();
        bool leaf();
};

#include "octree.cpp"
