namespace density
{

DEVICE_TARGET
inline static double smooth(double x)
{
    return 3 * x * x - 2 * x * x * x;
}

DEVICE_TARGET
inline static double smooth_d(double x)
{
    return 6 * x - 6 * x * x;
}

DEVICE_TARGET
inline static double linear_interpolate(double a, double b, double t)
{
    return (1. - t) * a + t * b;
}

DEVICE_TARGET
inline static F3::vec3_t get_grad(int x, int y, int z, size_t seed)
{
    constexpr int permutations[] =
    {
        151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,
        36,103,30,69,142,8,99,37,240,21,10,23,190,6,148,247,120,234,
        75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,88,237,
        149,56,87,174,20,125,136,171,168,68,175,74,165,71,134,139,
        48,27,166,77,146,158,231,83,111,229,122,60,211,133,230,220,
        105,92,41,55,46,245,40,244,102,143,54,65,25,63,161,1,216,80,
        73,209,76,132,187,208,89,18,169,200,196,135,130,116,188,159,
        86,164,100,109,198,173,186,3,64,52,217,226,250,124,123,5,
        202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,
        182,189,28,42,223,183,170,213,119,248,152,2,44,154,163,70,
        221,153,101,155,167,43,172,9,129,22,39,253,19,98,108,110,79,
        113,224,232,178,185,112,104,218,246,97,228,251,34,242,193,
        238,210,144,12,191,179,162,241,81,51,145,235,249,14,239,107,
        49,192,214,31,181,199,106,157,184,84,204,176,115,121,50,45,
        127,4,150,254,138,236,205,93,222,114,67,29,24,72,243,141,
        128,195,78,66,215,61,156,180
    };
    constexpr int grad[16][3] =
    {
        {1,1,0}, {-1,1,0}, {1,-1,0}, {-1,-1,0},
        {1,0,1}, {-1,0,1}, {1,0,-1}, {-1,0,-1},
        {0,1,1}, {0,-1,1}, {0,1,-1}, {0,-1,-1},
        {1,1,0}, {-1,1,0}, {0,-1,1}, {0,-1,-1}
    };
    int rand_value = permutations[(z * seed
        + permutations[(y * seed
        + permutations[(x * seed) & 255]) & 255]) & 255];
    int idx = rand_value & 15;
    return F3::vec3_t{grad[idx][0], grad[idx][1], grad[idx][2]};
}

DEVICE_TARGET
inline static F3 perlin(const F3::vec3_t& position,
        double normalization, size_t seed)
{
    double x = position.x / normalization;
    double y = position.y / normalization;
    double z = position.z / normalization;
    int X = x;
    int Y = y;
    int Z = z;
    x -= X;
    y -= Y;
    z -= Z;
    X &= 255;
    Y &= 255;
    Z &= 255;

    // Generate gradients
    const double g000 = glm::dot(get_grad(X, Y, Z, seed),
        F3::vec3_t(x, y, z));
    const double g001 = glm::dot(get_grad(X, Y, Z + 1, seed),
        F3::vec3_t(x, y, z - 1.));
    const double g010 = glm::dot(get_grad(X, Y + 1, Z, seed),
        F3::vec3_t(x, y - 1., z));
    const double g011 = glm::dot(get_grad(X, Y + 1, Z + 1, seed),
        F3::vec3_t(x, y - 1., z - 1.));
    const double g100 = glm::dot(get_grad(X + 1, Y, Z, seed),
        F3::vec3_t(x - 1., y, z));
    const double g101 = glm::dot(get_grad(X + 1, Y, Z + 1, seed),
        F3::vec3_t(x - 1., y, z - 1.));
    const double g110 = glm::dot(get_grad(X + 1, Y + 1, Z, seed),
        F3::vec3_t(x - 1., y - 1., z));
    const double g111 = glm::dot(get_grad(X + 1, Y + 1, Z + 1, seed),
            F3::vec3_t(x - 1., y - 1., z - 1.));

    // Smooth
    const double u = smooth(x);
    const double v = smooth(y);
    const double w = smooth(z);
    const double d_u = smooth_d(x);
    const double d_v = smooth_d(y);
    const double d_w = smooth_d(z);

    // Interpolate
    const double x00 = linear_interpolate(g000, g100, u);
    const double x10 = linear_interpolate(g010, g110, u);
    const double x01 = linear_interpolate(g001, g101, u);
    const double x11 = linear_interpolate(g011, g111, u);

    const double xy0 = linear_interpolate(x00, x10, v);
    const double xy1 = linear_interpolate(x01, x11, v);

    const double xyz = linear_interpolate(xy0, xy1, w);

    const double k1 = g100 - g000;
    const double k2 = g010 - g000;
    const double k3 = g001 - g000;
    const double k4 = g000 - g100 - g010 + g110;
    const double k5 = g000 - g010 - g001 + g011;
    const double k6 = g000 - g100 - g001 + g101;
    const double k7 = - g000 + g100 + g010 - g110 + g001 - g101 - g011 + g111;

    double d_x = 2.0 * d_u * (k1 + k4 * v + k6 * w + k7 * v * w);
    double d_y = 2.0 * d_v * (k2 + k5 * w + k4 * u + k7 * w * u);
    double d_z = 2.0 * d_w * (k3 + k6 * u + k5 * v + k7 * u * v);

    return F3{xyz, F3::vec3_t(d_x, d_y, d_z)};
}

}
