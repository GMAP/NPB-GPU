# NAS Parallel Benchmarks for GPU

This is a repository aimed at providing GPU parallel codes with different parallel APIs for the NAS Parallel Benchmarks ([NPB](https://www.nas.nasa.gov/publications/npb.html)) from a C/C++ version ([NPB-CPP](https://github.com/GMAP/NPB-CPP)). You can also contribute with this project, writing issues and pull requests. :smile:


:sound:*News:* Parametrization support for configuring number of threads per block and CUDA parallelism optimizations. :date:25/Jul/2021

:sound:*News:* CUDA versions for pseudo-applications added and IS improved. :date:11/Feb/2021

:sound:*News:* Paper published in the journal Software: Practice and Experience (SPE). :date:29/Nov/2021


## How to cite our work :+1:

[DOI](https://doi.org/10.1002/spe.3056) - Araujo, G.; Griebler, D.; Rockenbach, D. A.; Danelutto, M.; Fernandes, L. G.; **NAS Parallel Benchmarks with CUDA and beyond**, Software: Practice and Experience (SPE), 2021.

[DOI](https://doi.org/10.1109/PDP50117.2020.00009) - Araujo, G.; Griebler, D.; Danelutto, M.; Fernandes, L. G.; **Efficient NAS Benchmark Kernels with CUDA**. *28th Euromicro International Conference on Parallel, Distributed and Network-based Processing (PDP)*, Västerås, 2020. 
  
## The NPB with CUDA

The parallel CUDA version was implemented from the serial version of [NPB-CPP](https://github.com/GMAP/NPB-CPP).

==================================================================

NAS Parallel Benchmarks code contributors with CUDA are:

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

*Warning: our tests were made with GCC and CUDA*

## How to Compile


Go inside the directory `CUDA` directory and execute:

```
make _BENCHMARK CLASS=_VERSION
```

`_BENCHMARKs` are:


CG, EP, FT, IS, MG, BT, LU, and SP 


`_VERSIONs` are:

+ Class S: small for quick test purposes

+ Class W: workstation size (a 90's workstation; now likely too small)

+ Classes A, B, C: standard test problems; ~4X size increase going from one class to the next

+ Classes D, E, F: large test problems; ~16X size increase from each of the previous Classes


Command example:

```
make ep CLASS=B
```
  

## Activating the additional timers

NPB-GPU has additional timers for profiling purpose. To activate these timers, create a dummy file 'timer.flag' in the main directory of the NPB version (e.g. CUDA/timer.flag).

## Configuring the number of threads per block

NPB-GPU allows configuring the number of threads per block of each GPU kernel in the benchmarks. The user can specify the number of threads per block by editing the file gpu.config in the directory <API>/config/. If no file is specified, all GPU kernels are executed using the warp size of the GPU as the number of threads per block.

Syntax of the gpu.config file: 

```
<benchmark-name>_THREADS_PER_BLOCK_<gpu-kernel-name> = <interger-value>
```

Configuring CG benchmark as example:

```
CG_THREADS_PER_BLOCK_ON_KERNEL_ONE = 32
CG_THREADS_PER_BLOCK_ON_KERNEL_TWO = 128
CG_THREADS_PER_BLOCK_ON_KERNEL_THREE = 64
CG_THREADS_PER_BLOCK_ON_KERNEL_FOUR = 256
CG_THREADS_PER_BLOCK_ON_KERNEL_FIVE = 32
CG_THREADS_PER_BLOCK_ON_KERNEL_SIX = 64
CG_THREADS_PER_BLOCK_ON_KERNEL_SEVEN = 128
CG_THREADS_PER_BLOCK_ON_KERNEL_EIGHT = 64
CG_THREADS_PER_BLOCK_ON_KERNEL_NINE = 512
CG_THREADS_PER_BLOCK_ON_KERNEL_TEN = 512
CG_THREADS_PER_BLOCK_ON_KERNEL_ELEVEN = 1024
```

The NPB-GPU also allows changing the GPU device by providing the following syntax in the gpu.config file:

```
GPU_DEVICE = <interger-value>
```