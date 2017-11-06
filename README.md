### Dependencies:
- Cmake 3.0
- g++5
- Cuda8
- OpenGL4.5

### Build:
The two consecutive cmake and make are temporary. There are needed to compile
the generated files.
    $ mkdir build && cd build && cmake .. && make && cmake . && make

### Usage
All the commands below are already run by the second cmake && make
* `bin/make_dot` takes a .proc file and outputs a dot file of the AST.
    $ bin/make_dot examples/perlin.proc generated_files/perlin.dot && dot -Tsvg generated_files/perlin.dot > perlin.svg
* `compile2cuda` takes a .proc file and outputs a .cc file and a .cu file
    $ bin/compile2cuda examples/perlin.proc generated_files/perlin.cu  generated_files/perlin.cc

### Post-Compilation
The generated .cu and .cc files are compiled with src/app/main.cc and src/app/main.cu
