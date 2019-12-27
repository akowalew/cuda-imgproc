# cuda-imgproc

Studies project - CUDA implementation of some image processing algorithms.

It consists of `process-image` target. This is a simple app to process some images, which are read using OpenCV and then written to hard disk. 
Image processing algorithm consists of five phases:
 - splitting BGR into separate components
 - median filtering on each component
 - gaussian blurring on each component
 - histogram equalization on each component
 - merging B, G, R components into final image. 

Most of the algorithms should be written manually. Only image reading / writing is done using OpenCV. 

## Requirements

In order to build project, you will need following stuff:
 - `GCC` or `clang` compilers (with C++14/C++17 support) on Unices, or
 - `MSVC` on Windows
 - `CUDA Runtime`, tested on version V9.1.85 (default available in Ubuntu 18.04)
 - `CMake`, at least 3.16
 - `Conan`, at least 1.10.1
 - `OpenCV` - for reading/writing images and for reference implementation of the algorithms. Must be installed globally in the system
 - `Google Benchmark` - microbenchmarking library - to be installed via Conan or globally in the system
 - `cuda-samples` - handy utilities needed when developing for CUDA. Available via Git submodule

OpenCV and CUDA runtime is desired to be installed on host system. Other libraries may be managed using Conan package manager. 

## Compilation

First, you have to download Git submodules:

```sh
# Download git submodules
git submodule init
git submodule update
```

Then, in order to compile the project, create out-of-source build directory:

```sh
# Create out-of-source build directory
mkdir -p build
cd build/
```

Then, if you would like to use Conan package manager, type:

```sh
# Install Conan remotes configuration
conan config install ../conan

# Install Conan dependencies
conan install ../ --build=missing --setting compiler.libcxx=libstdc++11
```

Now you can configure build system:

```sh
# Configure build system
cmake ../ \
    -DCMAKE_BUILD_TYPE=<Release/Debug> \
    -DBUILD_TESTING=<ON/OFF> \
    -DBUILD_DEBUG=<ON/OFF> \
    -DBUILD_BENCHMARKING=<ON/OFF> \
    -DBUILD_CUDA=<ON/OFF> \
    -DBUILD_CONAN=<ON/OFF> \
    -DBUILD_VERSION=<reference/cpuserial/cpuparallel/cuda1/cuda2> \
```

In order to build everything, just type:

```sh
# Compile everything
make -j${nproc}
```

## Examples

There is only one app in the project: `process-image`. Its purpose is to process image at given path using `imgproc` library. Result will be printed on screen and saved to the output file:

```sh
./bin/process-image <input_file> <output_file>
```

Example: 

```sh
./bin/process-image ../assets/sample.jpg sample_out.jpg
```

## Benchmarking

Benchmarks are written using Google Benchmark library. Before you start any of the benchmark, make sure, that you've compiled project in Release version:

```sh
cmake ../ -DCMAKE_BUILD_TYPE=Release
```

Also, make sure, that you are able to use maximum power of your computer (e.g. ensure your laptop to be connected to the power source). Additionally, under the Linux, you may turn performance mode on:

```sh
# Enable performance mode
sudo cpupower frequency-set --governor performance
```

Then you can test your functions under benchmark suite. Example invokation:

```sh
./bin/process-image-bench --benchmark_repetitions=25
```

Since benchmark is written with Google Benchmark, you may take advantage of CLI options provided by it (just type `--help` to see all of them)

After the all, turn your computer back to powersave:

```sh
# Enable powersave mode
sudo cpupower frequency-set --governor powersave
```

## Tests

Units tests are written with [doctest](https://github.com/onqtam/doctest) library. In order to run all tests, just type:

```sh
make test
```

## Authors:

- [akowalew](https://github.com/akowalew)
- [Lasica](https://github.com/Lasica)
- [Lybba](https://github.com/lybba)
