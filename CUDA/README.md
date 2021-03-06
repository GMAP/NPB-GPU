# NAS Parallel Benchmarks for GPU

This is a repository aimed at providing GPU parallel codes with different parallel APIs for the NAS Parallel Benchmarks ([NPB](https://www.nas.nasa.gov/publications/npb.html)) from a C/C++ version ([NPB-CPP](https://github.com/GMAP/NPB-CPP)). You can also contribute with this project, writing issues and pull requests. :smile:


:sound:*News:* CUDA versions for pseudo-applications added and IS improved. :date:11/Feb/2021

## How to cite our work :+1:

[DOI](https://doi.org/10.1109/PDP50117.2020.00009) - Araujo, G. A.; Griebler, D.; Danelutto, M.; Fernandes, L. G. **Efficient NAS Benchmark Kernels with CUDA**. *28th Euromicro International Conference on Parallel, Distributed and Network-based Processing (PDP)*, Västerås, 2020. 
  
## The NPB with CUDA

The parallel CUDA version was implemented from the serial version of [NPB-CPP](https://github.com/GMAP/NPB-CPP).

==================================================================

NAS Parallel Benchmarks code contributors with CUDA are:

Dalvan Griebler: dalvan.griebler@acad.pucrs.br

Gabriell Araujo: gabriell.araujo@acad.pucrs.br

==================================================================

Each directory is independent and contains its own implemented version:

*Five kernels*

+ **IS** - Integer Sort, random memory access
+ **EP** - Embarrassingly Parallel
+ **CG** - Conjugate Gradient, irregular memory access and communication
+ **MG** - Multi-Grid on a sequence of meshes, long- and short-distance communication, memory intensive
+ **FT** - discrete 3D fast Fourier Transform, all-to-all communication

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


CG, EP, FT, IS, MG, SP, BT, and LU 


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

NPB3.3.1 includes additional timers for profiling purpose. To activate these timers, create a dummy file 'timer.flag' in the main directory of the NPB.

## Notes about the Fortran to CPP convertion

The following information are also written and keep updated in [NPB-CPP](https://github.com/GMAP/NPB-CPP):

+ Memory conventions adopted on NPB-CPP turn better the performance of the C++ code, lowering the execution time and memory consumption (on some applications, these conventions turn the performance of the NPB-CPP even better than the original Fortran NPB3.3.1, as for example on BT pseudo-application).  

  - Any global array is allocated with dynamic memory and as one single dimension.

  - On kernels, a cast is made in the functions, so is possible to work using multi-dimension accesses with the arrays, so, for example a function can receive an array like `matrix_aux[NY*NY]`, and work with accesses like `matrix_aux[j][i]`, instead of one single dimension access (actually, the cast in the functions follows the original NPB3.3.1 way).

  - On pseudo-applications, the cast is done already in the array declarations (NPB3.3.1 does not use one single dimension on pseudo-applications, so we cast the arrays directly on declarations, because this way, changes in the structure of the functions are not necessary).

  - To disable this convention (dynamic memory and on single dimension) is necessary set the flag `-DDO_NOT_ALLOCATE_ARRAYS_WITH_DYNAMIC_MEMORY_AND_AS_SINGLE_DIMENSION` on compilation.

  - Also, we keep every single structure of the loops as the same as Fortran original, but as Fortran memory behavior is different from C/C++, we invert the dimensions of the arrays and consequently, any array access. It is like change for example from an access `array[i][j][k]` to `array[k][j][i]` (but keeping the same organization of the loops), and change array dimensions from `array[NX][NY][NZ]` to `array[NZ][NY][NX]`.   
 
+ The original NPB has a file for print results of IS (`c_print_results.c`) and another file for print results of the other Benchmarks (`print_results.f`), it means, one file for Fortran code and one file for C code (IS is the only Benchmark that was written using C language). As the entire NPB-CPP is in C++, we keep only a single file to print results of all Benchmarks, we merged these two files and created `c_print_results.cpp`.

+ Any `goto` in the applications was replaced with an equivalent code using loops, keeping the same logic.

+ There are some little differences on indexes and ranges of loops, that are inherent to conversions from Fortran to C/C++.

+ In the file `common/npb-CPP.hpp` we define additional stuff like the structure and operations for complex numbers.

+ FT

	- Instead convert code directly from serial FT 3.3.1, we convert Fortran code from FT OpenMP version, where the format is equal to the FT serial versions before NPB 3.0.
	In no version of NPB the OpenMP parallel code is based on the serial code presented in 3.0 and 3.3.1 versions.
	The parallel code OpenMP in all versions of NPB is based on the sequential format before to the versions 3.0 and 3.3.1.
	In addition, in version 3.4, the most recent, they state that the sequential code will no longer be available. The sequential code will be the OpenMP code without compiling with OpenMP.
	Faced with these facts, we conclude that this refactored FT serial code (3.3.1) has no utility, as it was not used for parallel versions and from NPB 3.4 it will be completely forgotten.
  - In the global.hpp, historically, the constants `FFTBLOCK_DEFAULT` and `FFTBLOCKPAD_DEFAULT` receive values that change the cache behavior of the applications and the performance can be better or worse for each processor according which values are choosed. We define these constants with the value 1 (DEFAULT_BEHAVIOR), that determines a default behavior independently of the processor where the application is running.
  - The size of the matrixes on the original NPB is `[NZ][NY][NX+1]`, but we changed to `[NZ][NY][NX]`, because the additional positions generated by `NX+1` are not used on the application, they only spend more memory.
  - On the original NPB, the auxiliary matrixes `y1`, `y2` and `u` have the size as `NX`. But only in cffts1 the size is NX, on the cffts2 the correct size is NY and cffts3 the size is NZ. It is a problem when NX is not the bigger dimension. To fix this, we assign the size of these matrixes as `MAXDIM` that is the size of the bigger dimension. Consequently `MAXDIM` is also used as argument in the function `fft_init` that initializes the values of the matrix `u`.

+ IS
	- On the original NPB, IS has on its own source functions like `randlc`. On our version of NPB we does not need it, because we already have these functions implemented. On original NPB these functions are needed in the IS code, because IS is in C and these functions were written in Fortran for the other applications.
