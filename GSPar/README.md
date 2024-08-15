# NAS Parallel Benchmarks for GPU

This is a repository aimed at providing GPU parallel codes with different parallel APIs for the NAS Parallel Benchmarks ([NPB](https://www.nas.nasa.gov/publications/npb.html)) from a C/C++ version ([NPB-CPP](https://github.com/GMAP/NPB-CPP)). You can also contribute with this project, writing issues and pull requests. :smile:

:sound:*News:* CUDA versions for pseudo-applications added and IS improved. :date:11/Feb/2021

:sound:*News:* Parametrization support for configuring number of threads per block and CUDA parallelism optimizations. :date:25/Jul/2021

:sound:*News:* Paper published in the journal Software: Practice and Experience (SPE). :date:29/Nov/2021

:sound:*News:* A new GPU parallel implementation is now available using the [GSParLib API](https://github.com/GMAP/GSParLib). :date:15/Aug/2024


## How to cite our work :+1:

[DOI](https://doi.org/10.1002/spe.3056) - Araujo, G.; Griebler, D.; Rockenbach, D. A.; Danelutto, M.; Fernandes, L. G.; **NAS Parallel Benchmarks with CUDA and beyond**, Software: Practice and Experience (SPE), 2021.

[DOI](https://doi.org/10.1109/PDP50117.2020.00009) - Araujo, G.; Griebler, D.; Danelutto, M.; Fernandes, L. G.; **Efficient NAS Benchmark Kernels with CUDA**. *28th Euromicro International Conference on Parallel, Distributed and Network-based Processing (PDP)*, Västerås, 2020. 
  
## The NPB with GSParLib API

[GSParLib API](https://github.com/GMAP/GSParLib) is a C++ object-oriented multi-level API unifying OpenCL and CUDA for GPU programming that allows code portability between different GPU platforms and targets stream and data parallelism. GSParLib is organized into two APIs: 1) Driver API is a wrapper over CUDA and OpenCL; 2) Pattern API is a set of algorithmic skeletons.

The NPB parallel version using the GSParLib API was implemented from the serial version of [NPB-CPP](https://github.com/GMAP/NPB-CPP).

==================================================================

NAS Parallel Benchmarks code contributors with GSParLib API are:

Dalvan Griebler: dalvan.griebler@pucrs.br

Gabriell Araujo: gabriell.araujo@edu.pucrs.br

==================================================================

Each directory is independent and contains its own implemented version:

*Five kernels*

+ **IS** - Integer Sort
+ **EP** - Embarrassingly Parallel
+ **CG** - Conjugate Gradient
+ **MG** - Multi-Grid
+ **FT** - discrete 3D fast Fourier Transform

*Three pseudo-application*

+ **SP** - Scalar Penta-diagonal solver
+ **BT** - Block Tri-diagonal solver
+ **LU** - Lower-Upper Gauss-Seidel solver
  

## Software Requiriments

*Warning: our tests were made with G++ and CUDA*

## How to Compile and Run

1. Go inside the `GSPar` directory and execute the script `setup.sh` to download and install the GSParLib API:
    ```
    ./setup.sh
    ```

2. Go inside the `Driver` or `Pattern` directory and execute:

    ```
    make _BENCHMARK CLASS=_VERSION GPU_DRIVER=_GPU_BACKEND
    ```

    `_BENCHMARKs` are:


        CG, EP, FT, IS, MG, BT, LU, and SP 


    `_VERSIONs` are:

        + Class S: small for quick test purposes

        + Class W: workstation size (a 90's workstation; now likely too small)

        + Classes A, B, C: standard test problems; ~4X size increase going from one class to the next

        + Classes D, E, F: large test problems; ~16X size increase from each of the previous Classes
        
    `_GPU_BACKENDs` are:
    
        CUDA and OPENCL

    Command example:

    ```
    make ep CLASS=B GPU_DRIVER=CUDA
    ```
    
3. Export GSParLib path before executing the benchmark:

    ```
    export LD_LIBRARY_PATH=<NPB_GPU_PATH>/GSPar/lib/gspar/bin:$LD_LIBRARY_PATH
    ```