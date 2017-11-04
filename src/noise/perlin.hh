#include <iostream>
#include <algorithm>
#include <math.h>

class Perlin
{
    public:
        Perlin();
        void set_smoothing(std::function<double(double)> smoothing) {this-> smoothing = smoothing;};
        template<int normalization>
        double get_value(double x, double y, double z);
        double linear_interpolate(double a, double b, double t);
        double dot(const int *v, const double x, const double y, const double z);
        int *get_grad(int x, int y, int z);

    private:
        std::function<double(double)> smoothing;
        static const unsigned int permutations[512];
        static int grad[16][3];
};

#include "perlin.cc"
