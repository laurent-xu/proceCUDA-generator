#pragma once

#include <iostream>
#include <algorithm>
#include <math.h>
#include "noise.hh"


class Perlin: public Noise
{
    public:
        Perlin(float normalization = 100);
        void set_smoothing(std::function<int(int)> smoothing) {this-> smoothing = smoothing;};
        void set_normalization(float normalization) {this->normalization = normalization;};
        float getValue(float x, float y);
        void shuffle_permutations() {std::random_shuffle(std::begin(this->permutations), std::end(this->permutations));};

    private:
        float normalization;
        std::function<int(int)> smoothing;
        unsigned int permutations[256];
        static const float unit;
        static const float grad[][2];
};
