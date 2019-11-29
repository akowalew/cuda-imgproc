# cuda-imgproc

Studies project - CUDA implementation of some image processing algorithms.

It consists of two main targets:
 - `imgproc` library - The place where all calculations take place. At the moment, implemented as forwarding calls to OpenCV, but the target is to implement it using CUDA
 - `process-image` application - simple app which uses `imgproc` library to process some images, which are read using OpenCV.

## Requirements

In order to build project, you will need following stuff:
 - `GCC` or `clang` compilers (with C++14/C++17 support)
 - `CUDA Runtime`, tested on version V9.1.85 (available in Ubuntu 18.04)
 - `CMake`, at least 3.10
 - `OpenCV`
 - `Conan`, at least 1.10.1

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
