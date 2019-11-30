# cuda-imgproc

Studies project - CUDA implementation of some image processing algorithms.

It consists of `process-image` target. This is a simple app to process some images, which are read using OpenCV and then written to hard disk. 
Image processing algorithm consists of five phases:
 - splitting BGR i to separate components
 - median filtering on each component
 - gaussian blurring on each component
 - histogram equalization on each component
 - merging B, G, R components into final image. 

Most of the algorithms should be written manually. Only image reading / writing is done using OpenCV. 

## Requirements

In order to build project, you will need following stuff:
 - `GCC` or `clang` compilers (with C++14/C++17 support)
 - `CUDA Runtime`, tested on version V9.1.85 (available in Ubuntu 18.04)
 - `CMake`, at least 3.10
 - `OpenCV`
 - `Conan`, at least 1.10.1

OpenCV and CUDA runtime is desired to be installed on host system. Other libraries are managed using Conan package manager. 

## Compilation

In order to compile the project, type in terminal:

```sh
    # Create out-of-source build directory
    mkdir -p build
    cd build/

    # Download and install 3rd-party dependencies
    conan install ../

    # Configure build system
    cmake ../ \
    	-DCMAKE_BUILD_TYPE=<Release/Debug> \		# Build type optimization
    	-DBUILD_TESTING=<ON/OFF> \			# Whether to enable testing or not
    	-DIMGPROC_BUILD_TESTS=<ON/OFF> \		# Whether to build tests for imgproc or not

    # Compile everything
    make -j${nproc}
```

## Examples

There is only one app in the project: `process-image`. Its purpose is to process image at given path using `imgproc` library. Result will be printed on screen:

```sh
    ./bin/process-image ../assets/sample.jpg
```

## Tests

Units tests are written with [doctest](https://github.com/onqtam/doctest) library. In order to run all tests, just type:

```sh
	make test
```

To get more details about `imgproc` tests, run them directly:

```sh
	./bin/imgproc-test
```

## Authors:

- [akowalew](https://github.com/akowalew)
- [Lasica](https://github.com/Lasica)
