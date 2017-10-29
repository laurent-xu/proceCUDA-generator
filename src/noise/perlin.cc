#include "perlin.hh"

const float Perlin::unit = 1.0f / sqrt(2);
const float Perlin::grad[][2] =
{
    {Perlin::unit, Perlin::unit},
    {-Perlin::unit,Perlin::unit},
    {Perlin::unit,-Perlin::unit},
    {-Perlin::unit,-Perlin::unit},
    {1,0},
    {-1,0},
    {0,1},
    {0,-1}
};

Perlin::Perlin(float normalization) : normalization(normalization),
    smoothing([](int x) -> int {return 3 * x * x - 2 * x * x * x;})
{
    for (int i = 0; i < 256; ++i)
        this->permutations[i] = i;
    this->shuffle_permutations();
}

float Perlin::getValue(float x, float y)
{
    float tempX, tempY;
    int x0, y0, ii, jj, gi0, gi1, gi2, gi3;
    float tmp, s, t, u, v, Cx, Cy, Li1, Li2;
    x /= this->normalization;
    y /= this->normalization;
    x0 = (int)(x);
    y0 = (int)(y);
    ii = x0 & 255;
    jj = y0 & 255;
    gi0 = this->permutations[ii + this->permutations[jj]] % 8;
    gi1 = this->permutations[ii + 1 + this->permutations[jj]] % 8;
    gi2 = this->permutations[ii + permutations[jj + 1]] % 8;
    gi3 = this->permutations[ii + 1 + this->permutations[jj + 1]] % 8;

    // Generate gradients
    tempX = x - x0;
    tempY = y - y0;
    s = this->grad[gi0][0] * tempX + this->grad[gi0][1] * tempY;

    tempX = x - (x0 + 1);
    tempY = y - y0;
    t = this->grad[gi1][0] * tempX + this->grad[gi1][1] * tempY;

    tempX = x - x0;
    tempY = y - (y0 + 1);
    u = this->grad[gi2][0] * tempX + this->grad[gi2][1] * tempY;

    tempX = x - (x0 + 1);
    tempY = y - (y0 + 1);
    v = this->grad[gi3][0] * tempX + this->grad[gi3][1] * tempY;

    // Smooth and interpolate
    tmp = x - x0;
    Cx = smoothing(tmp);

    Li1 = s + Cx * (t - s);
    Li2 = u + Cx * (v - u);

    tmp = y - y0;
    Cy = smoothing(tmp);

    return Li1 + Cy * (Li2 - Li1);
}
