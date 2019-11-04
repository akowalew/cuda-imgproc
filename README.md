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

## Compilation

In order to compile the project, type in terminal:

```sh
    mkdir -p build
    cd build/
    cmake ../ -DCMAKE_BUILD_TYPE=Release
    make -j${nproc}
```

## Usage

There is only one app in the project: `process-image`. Its purpose is to process image at given path using `imgproc` library. Result will be printed on screen:

```sh
    ./process-image/process-image ../assets/sample.jpg
```

## Authors:

- [akowalew](https://github.com/akowalew)
- [Lasica](https://github.com/Lasica)
