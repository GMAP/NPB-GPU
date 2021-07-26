/* 
 * ------------------------------------------------------------------------------
 *
 * MIT License
 *
 * Copyright (c) 2021 Parallel Applications Modelling Group - GMAP
 *      GMAP website: https://gmap.pucrs.br
 *
 * Pontifical Catholic University of Rio Grande do Sul (PUCRS)
 * Av. Ipiranga, 6681, Porto Alegre - Brazil, 90619-900
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * ------------------------------------------------------------------------------
 *
 * The original NPB 3.4 version was written in Fortran and belongs to:
 *      http://www.nas.nasa.gov/Software/NPB/
 *
 * Authors of the Fortran code:
 *      M. Yarrow
 *      H. Jin
 *
 * ------------------------------------------------------------------------------
 *
 * The serial C++ version is a translation of the original NPB 3.4
 * Serial C++ version: https://github.com/GMAP/NPB-CPP/tree/master/NPB-SER
 *
 * Authors of the C++ code:
 *      Dalvan Griebler <dalvangriebler@gmail.com>
 *      Gabriell Araujo <hexenoften@gmail.com>
 *      Júnior Löff <loffjh@gmail.com>
 *
 * ------------------------------------------------------------------------------
 *
 * The CUDA version is a parallel implementation of the serial C++ version
 * CUDA version: https://github.com/GMAP/NPB-GPU/tree/master/CUDA
 *
 * Authors of the CUDA code:
 *      Gabriell Araujo <hexenoften@gmail.com>
 *
 * ------------------------------------------------------------------------------
 */

#include <cuda.h>
#include "../common/npb-CPP.hpp"
#include "npbparams.hpp"

#define PROFILING_TOTAL_TIME (0)
#define PROFILING_CREATE (1)
#define PROFILING_RANK (2)
#define PROFILING_VERIFY (3)

/*****************************************************************/
/* for serial IS, buckets are not really req'd to solve NPB1 IS  */
/* spec, but their use on some machines improves performance, on */
/* other machines the use of buckets compromises performance,    */
/* probably because it is extra computation which is not req'd.  */
/* (note: mechanism not understood, probably cache related)      */
/* example: SP2-66MhzWN: 50% speedup with buckets                */
/* example: SGI Indy5000: 50% slowdown with buckets              */
/* example: SGI O2000: 400% slowdown with buckets (Wow!)         */
/*****************************************************************/
/* to disable the use of buckets, comment out the following line */
#define USE_BUCKETS

/******************/
/* default values */
/******************/
#ifndef CLASS
#define CLASS 'S'
#endif

/*************/
/*  CLASS S  */
/*************/
#if CLASS == 'S'
#define TOTAL_KEYS_LOG_2 (16)
#define MAX_KEY_LOG_2 (11)
#define NUM_BUCKETS_LOG_2 (9)
#endif

/*************/
/*  CLASS W  */
/*************/
#if CLASS == 'W'
#define TOTAL_KEYS_LOG_2 (20)
#define MAX_KEY_LOG_2 (16)
#define NUM_BUCKETS_LOG_2 (10)
#endif

/*************/
/*  CLASS A  */
/*************/
#if CLASS == 'A'
#define TOTAL_KEYS_LOG_2 (23)
#define MAX_KEY_LOG_2 (19)
#define NUM_BUCKETS_LOG_2 (10)
#endif

/*************/
/*  CLASS B  */
/*************/
#if CLASS == 'B'
#define TOTAL_KEYS_LOG_2 (25)
#define MAX_KEY_LOG_2 (21)
#define NUM_BUCKETS_LOG_2 (10)
#endif

/*************/
/*  CLASS C  */
/*************/
#if CLASS == 'C'
#define TOTAL_KEYS_LOG_2 (27)
#define MAX_KEY_LOG_2 (23)
#define NUM_BUCKETS_LOG_2 (10)
#endif

/*************/
/*  CLASS D  */
/*************/
#if CLASS == 'D'
#define TOTAL_KEYS_LOG_2 (31)
#define MAX_KEY_LOG_2 (27)
#define NUM_BUCKETS_LOG_2 (10)
#endif

#if CLASS == 'D'
#define TOTAL_KEYS (1L << TOTAL_KEYS_LOG_2)
#else
#define TOTAL_KEYS (1 << TOTAL_KEYS_LOG_2)
#endif
#define MAX_KEY (1 << MAX_KEY_LOG_2)
#define NUM_BUCKETS (1 << NUM_BUCKETS_LOG_2)
#define NUM_KEYS (TOTAL_KEYS)
#define SIZE_OF_BUFFERS (NUM_KEYS)

#define MAX_ITERATIONS (10)
#define TEST_ARRAY_SIZE (5)

/*************************************/
/* typedef: if necessary, change the */
/* size of int here by changing the  */
/* int type to, say, long            */
/*************************************/
#if CLASS == 'D'
/* #TODO *//* is necessary implement INT_TYPE for class D */
/* typedef long INT_TYPE; */
#else
/* typedef int INT_TYPE; */
#endif

/**********************/
/* partial verif info */
/**********************/
int test_index_array[TEST_ARRAY_SIZE],
    test_rank_array[TEST_ARRAY_SIZE],

    S_test_index_array[TEST_ARRAY_SIZE] = 
{48427,17148,23627,62548,4431},
	S_test_rank_array[TEST_ARRAY_SIZE] = 
{0,18,346,64917,65463},

	W_test_index_array[TEST_ARRAY_SIZE] = 
{357773,934767,875723,898999,404505},
	W_test_rank_array[TEST_ARRAY_SIZE] = 
{1249,11698,1039987,1043896,1048018},

	A_test_index_array[TEST_ARRAY_SIZE] = 
{2112377,662041,5336171,3642833,4250760},
	A_test_rank_array[TEST_ARRAY_SIZE] = 
{104,17523,123928,8288932,8388264},

	B_test_index_array[TEST_ARRAY_SIZE] = 
{41869,812306,5102857,18232239,26860214},
	B_test_rank_array[TEST_ARRAY_SIZE] = 
{33422937,10244,59149,33135281,99}, 

	C_test_index_array[TEST_ARRAY_SIZE] = 
{44172927,72999161,74326391,129606274,21736814},
	C_test_rank_array[TEST_ARRAY_SIZE] = 
{61147,882988,266290,133997595,133525895},

	D_test_index_array[TEST_ARRAY_SIZE] = 
{1317351170,995930646,1157283250,1503301535,1453734525},
	D_test_rank_array[TEST_ARRAY_SIZE] = 
{1,36538729,1978098519,2145192618,2147425337};

/* global variables */
int passed_verification;
int timer_on;
int* key_array_device; 
int* key_buff1_device; 
int* key_buff2_device;
int* index_array_device; 
int* rank_array_device;
int* partial_verify_vals_device;
int* passed_verification_device;
int* key_scan_device; 
int* sum_device;
size_t size_test_array_device;
size_t size_key_array_device; 
size_t size_key_buff1_device; 
size_t size_key_buff2_device;
size_t size_index_array_device; 
size_t size_rank_array_device;
size_t size_partial_verify_vals_device;
size_t size_passed_verification_device;
size_t size_key_scan_device; 
size_t size_sum_device;
size_t size_shared_data_on_rank_4;
size_t size_shared_data_on_rank_5;
size_t size_shared_data_on_full_verify_3;
int threads_per_block_on_create_seq;
int threads_per_block_on_rank;
int threads_per_block_on_rank_1;
int threads_per_block_on_rank_2;
int threads_per_block_on_rank_3;
int threads_per_block_on_rank_4;
int threads_per_block_on_rank_5;
int threads_per_block_on_rank_6;
int threads_per_block_on_rank_7;
int threads_per_block_on_full_verify;
int threads_per_block_on_full_verify_1;
int threads_per_block_on_full_verify_2;
int threads_per_block_on_full_verify_3;
int blocks_per_grid_on_create_seq;
int blocks_per_grid_on_rank_1;
int blocks_per_grid_on_rank_2;
int blocks_per_grid_on_rank_3;
int blocks_per_grid_on_rank_4;
int blocks_per_grid_on_rank_5;
int blocks_per_grid_on_rank_6;
int blocks_per_grid_on_rank_7;
int blocks_per_grid_on_full_verify_1;
int blocks_per_grid_on_full_verify_2;
int blocks_per_grid_on_full_verify_3;
int amount_of_work_on_create_seq;
int amount_of_work_on_rank_1;
int amount_of_work_on_rank_2;
int amount_of_work_on_rank_3;
int amount_of_work_on_rank_4;
int amount_of_work_on_rank_5;
int amount_of_work_on_rank_6;
int amount_of_work_on_rank_7;
int amount_of_work_on_full_verify_1;
int amount_of_work_on_full_verify_2;
int amount_of_work_on_full_verify_3;
int gpu_device_id;
int total_devices;
cudaDeviceProp gpu_device_properties;
extern __shared__ int extern_share_data[];

/* function declarations */
static void create_seq_gpu(double seed, 
		double a);
__global__ void create_seq_gpu_kernel(int* key_array,
		double seed,
		double a,
		int number_of_blocks,
		int amount_of_work);
__device__ double find_my_seed_device(int kn,
		int np,
		long nn,
		double s,
		double a);
static void full_verify_gpu();
__global__ void full_verify_gpu_kernel_1(int* key_array,
		int* key_buff2,
		int number_of_blocks,
		int amount_of_work);
__global__ void full_verify_gpu_kernel_2(int* key_buff2,
		int* key_buff_ptr_global,
		int* key_array,
		int number_of_blocks,
		int amount_of_work);
__global__ void full_verify_gpu_kernel_3(int* key_array,
		int* global_aux,
		int number_of_blocks,
		int amount_of_work);
__device__ double randlc_device(double* X,
		double* A);
static void rank_gpu(int iteration);
__global__ void rank_gpu_kernel_1(int* key_array,
		int* partial_verify_vals,
		int* test_index_array,
		int iteration,
		int number_of_blocks,
		int amount_of_work);
__global__ void rank_gpu_kernel_2(int* key_buff1,
		int number_of_blocks,
		int amount_of_work);
__global__ void rank_gpu_kernel_3(int* key_buff_ptr,
		int* key_buff_ptr2,
		int number_of_blocks,
		int amount_of_work);
__global__ void rank_gpu_kernel_4(int* source,
		int* destiny,
		int* sum,
		int number_of_blocks,
		int amount_of_work);
__global__ void rank_gpu_kernel_5(int* source,
		int* destiny,
		int number_of_blocks,
		int amount_of_work);
__global__ void rank_gpu_kernel_6(int* source,
		int* destiny,
		int* offset,
		int number_of_blocks,
		int amount_of_work);
__global__ void rank_gpu_kernel_7(int* partial_verify_vals,
		int* key_buff_ptr,
		int* test_rank_array,
		int* passed_verification_device,
		int iteration,
		int number_of_blocks,
		int amount_of_work);
static void release_gpu();
static void setup_gpu();

/* is */
int main(int argc, char** argv){
#if defined(DO_NOT_ALLOCATE_ARRAYS_WITH_DYNAMIC_MEMORY_AND_AS_SINGLE_DIMENSION)
	printf(" DO_NOT_ALLOCATE_ARRAYS_WITH_DYNAMIC_MEMORY_AND_AS_SINGLE_DIMENSION mode on\n");
#endif
#if defined(PROFILING)
	printf(" PROFILING mode on\n");
#endif
	int i, iteration;
	double timecounter;

	timer_clear(PROFILING_TOTAL_TIME);
#if defined(PROFILING)
	timer_clear(PROFILING_CREATE);
	timer_clear(PROFILING_RANK);
	timer_clear(PROFILING_VERIFY);
#endif

#if defined(PROFILING)
	timer_start(PROFILING_TOTAL_TIME);
#endif

	/* initialize the verification arrays if a valid class */
	for(i=0; i<TEST_ARRAY_SIZE; i++){
		switch(CLASS){
			case 'S':
				test_index_array[i] = S_test_index_array[i];
				test_rank_array[i]  = S_test_rank_array[i];
				break;
			case 'A':
				test_index_array[i] = A_test_index_array[i];
				test_rank_array[i]  = A_test_rank_array[i];
				break;
			case 'W':
				test_index_array[i] = W_test_index_array[i];
				test_rank_array[i]  = W_test_rank_array[i];
				break;
			case 'B':
				test_index_array[i] = B_test_index_array[i];
				test_rank_array[i]  = B_test_rank_array[i];
				break;
			case 'C':
				test_index_array[i] = C_test_index_array[i];
				test_rank_array[i]  = C_test_rank_array[i];
				break;
			case 'D':
				test_index_array[i] = D_test_index_array[i];
				test_rank_array[i]  = D_test_rank_array[i];
				break;
		};
	}

	/* printout initial NPB info */
	printf("\n\n NAS Parallel Benchmarks 4.1 CUDA C++ version - IS Benchmark\n\n");
	printf(" Size:  %ld  (class %c)\n", (long)TOTAL_KEYS, CLASS);
	printf(" Iterations:   %d\n", MAX_ITERATIONS);

	setup_gpu();

#if defined(PROFILING)
	timer_start(PROFILING_CREATE);
#endif
	/* generate random number sequence and subsequent keys on all procs */
	create_seq_gpu(314159265.00, /* random number gen seed */
			1220703125.00); /* random number gen mult */
#if defined(PROFILING)
	timer_stop(PROFILING_CREATE);
#endif

	/* 
	 * do one interation for free (i.e., untimed) to guarantee initialization of  
	 * all data and code pages and respective tables 
	 */
	rank_gpu(1);  

	/* start verification counter */
	passed_verification = 0;

	cudaMemcpy(passed_verification_device, &passed_verification, size_passed_verification_device, cudaMemcpyHostToDevice);

	if(CLASS != 'S')printf( "\n   iteration\n");

#if defined(PROFILING)
	timer_start(PROFILING_RANK);
#else
	timer_start(PROFILING_TOTAL_TIME);
#endif
	/* this is the main iteration */
	for(iteration=1; iteration<=MAX_ITERATIONS; iteration++){
		if(CLASS != 'S')printf( "        %d\n", iteration);
		rank_gpu(iteration);
	}
#if defined(PROFILING)
	timer_stop(PROFILING_RANK);
#else
	timer_stop(PROFILING_TOTAL_TIME);
#endif

	cudaMemcpy(&passed_verification, passed_verification_device, size_passed_verification_device, cudaMemcpyDeviceToHost);	

	/* 
	 * this tests that keys are in sequence: sorting of last ranked key seq
	 * occurs here, but is an untimed operation                             
	 */
#if defined(PROFILING)
	timer_start(PROFILING_VERIFY);
#endif
	full_verify_gpu();
#if defined(PROFILING)
	timer_stop(PROFILING_VERIFY);
#endif
#if defined(PROFILING)
	timer_stop(PROFILING_TOTAL_TIME);
	timecounter = timer_read(PROFILING_RANK);
#else
	timecounter = timer_read(PROFILING_TOTAL_TIME);
#endif


	char gpu_config[256];
	char gpu_config_string[2048];
#if defined(PROFILING)
	sprintf(gpu_config, "%5s\t%25s\t%25s\t%25s\n", "GPU Kernel", "Threads Per Block", "Time in Seconds", "Time in Percentage");
	strcpy(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " create", threads_per_block_on_create_seq, timer_read(PROFILING_CREATE), (timer_read(PROFILING_CREATE)*100/timer_read(PROFILING_TOTAL_TIME)));
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " rank", threads_per_block_on_rank, timer_read(PROFILING_RANK), (timer_read(PROFILING_RANK)*100/timer_read(PROFILING_TOTAL_TIME)));
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " verify", threads_per_block_on_full_verify, timer_read(PROFILING_VERIFY), (timer_read(PROFILING_VERIFY)*100/timer_read(PROFILING_TOTAL_TIME)));
	strcat(gpu_config_string, gpu_config);
#else
	sprintf(gpu_config, "%5s\t%25s\n", "GPU Kernel", "Threads Per Block");
	strcpy(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\n", " create", threads_per_block_on_create_seq);
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\n", " rank", threads_per_block_on_rank);
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\n", " verify", threads_per_block_on_full_verify);
	strcat(gpu_config_string, gpu_config);
#endif

	/* the final printout  */
	if(passed_verification != 5*MAX_ITERATIONS+1){passed_verification = 0;}
	c_print_results((char*)"IS",
			CLASS,
			(int)(TOTAL_KEYS/64),
			64,
			0,
			MAX_ITERATIONS,
			timecounter,
			((double)(MAX_ITERATIONS*TOTAL_KEYS))/timecounter/1000000.0,
			(char*)"keys ranked",
			passed_verification,
			(char*)NPBVERSION,
			(char*)COMPILETIME,
			(char*)COMPILERVERSION,
			(char*)LIBVERSION,
			(char*)CPU_MODEL,
			(char*)gpu_device_properties.name,
			(char*)gpu_config_string,
			(char*)CS1,
			(char*)CS2,
			(char*)CS3,
			(char*)CS4,
			(char*)CS5,
			(char*)CS6,
			(char*)CS7);

	/* print additional timers */
	/*
	   if(timer_on){
	   double t_total, t_percent;
	   t_total = timer_read(TOTAL_EXECUTION);
	   printf("\nAdditional timers -\n");
	   printf(" Total execution: %8.3f\n", t_total);
	   if(t_total == 0.0)t_total = 1.0;
	   timecounter = timer_read(INITIALIZATION);
	   t_percent = timecounter/t_total * 100.;
	   printf(" Initialization : %8.3f (%5.2f%%)\n", timecounter, t_percent);
	   timecounter = timer_read(BENCHMARKING);
	   t_percent = timecounter/t_total * 100.;
	   printf(" Benchmarking   : %8.3f (%5.2f%%)\n", timecounter, t_percent);
	   timecounter = timer_read(SORTING);
	   t_percent = timecounter/t_total * 100.;
	   printf(" Sorting        : %8.3f (%5.2f%%)\n", timecounter, t_percent);
	   }
	 */

	release_gpu();

	return 0;  
}

static void create_seq_gpu(double seed, double a){  
	create_seq_gpu_kernel<<<blocks_per_grid_on_create_seq, 
		threads_per_block_on_create_seq>>>(key_array_device,
				seed,
				a,
				blocks_per_grid_on_create_seq,
				amount_of_work_on_create_seq);
	cudaDeviceSynchronize();
}

__global__ void create_seq_gpu_kernel(int* key_array,
		double seed,
		double a,
		int number_of_blocks,
		int amount_of_work){
	double x, s;
	int i, k;

	int k1, k2;
	double an = a;
	int myid, num_procs;
	int mq;

	myid = blockIdx.x*blockDim.x+threadIdx.x;
	num_procs = amount_of_work;

	mq = (NUM_KEYS + num_procs - 1) / num_procs;
	k1 = mq * myid;
	k2 = k1 + mq;
	if(k2 > NUM_KEYS){k2 = NUM_KEYS;}

	s = find_my_seed_device(myid, num_procs, (long)4*NUM_KEYS, seed, an);

	k = MAX_KEY/4;

	for(i=k1; i<k2; i++){
		x = randlc_device(&s, &an);
		x += randlc_device(&s, &an);
		x += randlc_device(&s, &an);
		x += randlc_device(&s, &an);  
		key_array[i] = k*x;
	}
}

__device__ double find_my_seed_device(int kn,
		int np,
		long nn,
		double s,
		double a){
	double t1,t2;
	long mq,nq,kk,ik;

	if(kn==0){return s;}

	mq = (nn/4 + np - 1) / np;
	nq = mq * 4 * kn;

	t1 = s;
	t2 = a;
	kk = nq;
	while(kk > 1){
		ik = kk / 2;
		if(2*ik==kk){
			(void)randlc_device(&t2, &t2);
			kk = ik;
		}else{
			(void)randlc_device(&t1, &t2);
			kk = kk - 1;
		}
	}
	(void)randlc_device(&t1, &t2);

	return(t1);
}

static void full_verify_gpu(){		
	int* memory_aux_device;
	size_t size_memory_aux=sizeof(int)*(amount_of_work_on_full_verify_3/threads_per_block_on_full_verify_3);		
	cudaMalloc(&memory_aux_device, size_memory_aux);	

	/* full_verify_gpu_kernel_1 */
	full_verify_gpu_kernel_1<<<blocks_per_grid_on_full_verify_1, 
		threads_per_block_on_full_verify_1>>>(key_array_device,
				key_buff2_device,
				blocks_per_grid_on_full_verify_1,
				amount_of_work_on_full_verify_1);
	cudaDeviceSynchronize();

	/* full_verify_gpu_kernel_2 */
	full_verify_gpu_kernel_2<<<blocks_per_grid_on_full_verify_2, 
		threads_per_block_on_full_verify_2>>>(key_buff2_device,
				key_buff1_device,
				key_array_device,
				blocks_per_grid_on_full_verify_2,
				amount_of_work_on_full_verify_2);
	cudaDeviceSynchronize();

	/* full_verify_gpu_kernel_3 */
	full_verify_gpu_kernel_3<<<blocks_per_grid_on_full_verify_3, 
		threads_per_block_on_full_verify_3,
		size_shared_data_on_full_verify_3>>>(key_array_device,
				memory_aux_device,
				blocks_per_grid_on_full_verify_3,
				amount_of_work_on_full_verify_3);
	cudaDeviceSynchronize();

	/* reduce on cpu */
	int i, j = 0;
	int* memory_aux_host=(int*)malloc(size_memory_aux);
	cudaMemcpy(memory_aux_host, memory_aux_device, size_memory_aux, cudaMemcpyDeviceToHost);
	for(i=0; i<size_memory_aux/sizeof(int); i++){
		j += memory_aux_host[i];
	}	

	if(j!=0){
		printf( "Full_verify: number of keys out of sort: %ld\n", (long)j );
	}else{
		passed_verification++;
	}

	cudaFree(memory_aux_device);
	free(memory_aux_host);
}

__global__ void full_verify_gpu_kernel_1(int* key_array,
		int* key_buff2,
		int number_of_blocks,
		int amount_of_work){
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	key_buff2[i] = key_array[i];
}

__global__ void full_verify_gpu_kernel_2(int* key_buff2,
		int* key_buff_ptr_global,
		int* key_array,
		int number_of_blocks,
		int amount_of_work){		
	int value = key_buff2[blockIdx.x*blockDim.x+threadIdx.x];
	int index = atomicAdd(&key_buff_ptr_global[value], -1) - 1;
	key_array[index] = value;
}

__global__ void full_verify_gpu_kernel_3(int* key_array,
		int* global_aux,
		int number_of_blocks,
		int amount_of_work){
	int* shared_aux = (int*)(extern_share_data);

	int i = (blockIdx.x*blockDim.x+threadIdx.x) + 1;

	if(i<NUM_KEYS){
		if(key_array[i-1]>key_array[i]){shared_aux[threadIdx.x]=1;}
		else{shared_aux[threadIdx.x]=0;}
	}else{shared_aux[threadIdx.x]=0;}

	__syncthreads();

	for(i=blockDim.x/2; i>0; i>>=1){
		if(threadIdx.x<i){
			shared_aux[threadIdx.x] += shared_aux[threadIdx.x+i];
		}
		__syncthreads();
	}

	if(threadIdx.x==0){global_aux[blockIdx.x]=shared_aux[0];}
}

__device__ double randlc_device(double* X,
		double* A){
	double T1, T2, T3, T4;
	double A1;
	double A2;
	double X1;
	double X2;
	double Z;
	int j;

	/*
	 * --------------------------------------------------------------------
	 * break A into two parts such that A = 2^23 * A1 + A2 and set X = N.
	 * --------------------------------------------------------------------
	 */
	T1 = R23 * *A;
	j  = T1;
	A1 = j;
	A2 = *A - T23 * A1;

	/*
	 * --------------------------------------------------------------------
	 * break X into two parts such that X = 2^23 * X1 + X2, compute
	 * Z = A1 * X2 + A2 * X1  (mod 2^23), and then
	 * X = 2^23 * Z + A2 * X2  (mod 2^46). 
	 * --------------------------------------------------------------------
	 */
	T1 = R23 * *X;
	j  = T1;
	X1 = j;
	X2 = *X - T23 * X1;
	T1 = A1 * X2 + A2 * X1;

	j  = R23 * T1;
	T2 = j;
	Z = T1 - T23 * T2;
	T3 = T23 * Z + A2 * X2;
	j  = R46 * T3;
	T4 = j;
	*X = T3 - T46 * T4;

	return(R46 * *X);
} 

static void rank_gpu(int iteration){
	/* rank_gpu_kernel_1 */
	rank_gpu_kernel_1<<<blocks_per_grid_on_rank_1, 
		threads_per_block_on_rank_1>>>(key_array_device,
				partial_verify_vals_device,
				index_array_device,
				iteration,
				blocks_per_grid_on_rank_1,
				amount_of_work_on_rank_1);

	/* rank_gpu_kernel_2 */
	rank_gpu_kernel_2<<<blocks_per_grid_on_rank_2, 
		threads_per_block_on_rank_2>>>(key_buff1_device,
				blocks_per_grid_on_rank_2,
				amount_of_work_on_rank_2);

	/* rank_gpu_kernel_3 */
	rank_gpu_kernel_3<<<blocks_per_grid_on_rank_3, 
		threads_per_block_on_rank_3>>>(key_buff1_device,
				key_array_device,
				blocks_per_grid_on_rank_3,
				amount_of_work_on_rank_3);

	/* rank_gpu_kernel_4 */
	rank_gpu_kernel_4<<<blocks_per_grid_on_rank_4, 
		threads_per_block_on_rank_4,
		size_shared_data_on_rank_4>>>(key_buff1_device,
				key_buff1_device,
				sum_device,
				blocks_per_grid_on_rank_4,
				amount_of_work_on_rank_4);

	/* rank_gpu_kernel_5 */
	rank_gpu_kernel_5<<<blocks_per_grid_on_rank_5, 
		threads_per_block_on_rank_5,
		size_shared_data_on_rank_5>>>(sum_device,
				sum_device,
				blocks_per_grid_on_rank_5,
				amount_of_work_on_rank_5);

	/* rank_gpu_kernel_6 */
	rank_gpu_kernel_6<<<blocks_per_grid_on_rank_6, 
		threads_per_block_on_rank_6>>>(key_buff1_device,
				key_buff1_device,
				sum_device,
				blocks_per_grid_on_rank_6,
				amount_of_work_on_rank_6);

	/* rank_gpu_kernel_7 */
	rank_gpu_kernel_7<<<blocks_per_grid_on_rank_7, 
		threads_per_block_on_rank_7>>>(partial_verify_vals_device,
				key_buff1_device,
				rank_array_device,
				passed_verification_device,
				iteration,
				blocks_per_grid_on_rank_7,
				amount_of_work_on_rank_7);
}

__global__ void rank_gpu_kernel_1(int* key_array,
		int* partial_verify_vals,
		int* test_index_array,
		int iteration,
		int number_of_blocks,
		int amount_of_work){
	key_array[iteration] = iteration;
	key_array[iteration+MAX_ITERATIONS] = MAX_KEY - iteration;
	/*
	 * --------------------------------------------------------------------
	 * determine where the partial verify test keys are, 
	 * --------------------------------------------------------------------
	 * load into top of array bucket_size  
	 * --------------------------------------------------------------------
	 */
#pragma unroll
	for(int i=0; i<TEST_ARRAY_SIZE; i++){
		partial_verify_vals[i] = key_array[test_index_array[i]];
	}
}

__global__ void rank_gpu_kernel_2(int* key_buff1,
		int number_of_blocks,
		int amount_of_work){
	key_buff1[blockIdx.x*blockDim.x+threadIdx.x] = 0;
}

__global__ void rank_gpu_kernel_3(int* key_buff_ptr,
		int* key_buff_ptr2,
		int number_of_blocks,
		int amount_of_work){
	/*
	 * --------------------------------------------------------------------
	 * in this section, the keys themselves are used as their 
	 * own indexes to determine how many of each there are: their
	 * individual population  
	 * --------------------------------------------------------------------
	 */
	atomicAdd(&key_buff_ptr[key_buff_ptr2[blockIdx.x*blockDim.x+threadIdx.x]], 1);
}

__global__ void rank_gpu_kernel_4(int* source,
		int* destiny,
		int* sum,
		int number_of_blocks,
		int amount_of_work){
	int* shared_data = (int*)(extern_share_data);

	shared_data[threadIdx.x] = 0;
	int position = blockDim.x + threadIdx.x;

	int factor = MAX_KEY / number_of_blocks;
	int start = factor * blockIdx.x;
	int end = start + factor;

	for(int i=start; i<end; i+=blockDim.x){
		shared_data[position] = source[i + threadIdx.x];

		for(uint offset=1; offset<blockDim.x; offset<<=1){
			__syncthreads();
			int t = shared_data[position] + shared_data[position - offset];
			__syncthreads();
			shared_data[position] = t;
		}

		int prv_val = (i == start) ? 0 : destiny[i - 1];
		destiny[i + threadIdx.x] = shared_data[position] + prv_val;
	}

	__syncthreads();
	if(threadIdx.x==0){sum[blockIdx.x]=destiny[end-1];}
}

__global__ void rank_gpu_kernel_5(int* source,
		int* destiny,
		int number_of_blocks,
		int amount_of_work){
	int* shared_data = (int*)(extern_share_data);

	shared_data[threadIdx.x] = 0;
	int position = blockDim.x + threadIdx.x;
	shared_data[position] = source[threadIdx.x];

	for(uint offset=1; offset<blockDim.x; offset<<=1){
		__syncthreads();
		int t = shared_data[position] + shared_data[position - offset];
		__syncthreads();
		shared_data[position] = t;
	}

	__syncthreads();

	destiny[threadIdx.x] = shared_data[position - 1];
}

__global__ void rank_gpu_kernel_6(int* source,
		int* destiny,
		int* offset,
		int number_of_blocks,
		int amount_of_work){
	int factor = MAX_KEY / number_of_blocks;
	int start = factor * blockIdx.x;
	int end = start + factor;
	int sum = offset[blockIdx.x];
	for(int i=start; i<end; i+=blockDim.x){
		destiny[i + threadIdx.x] = source[i + threadIdx.x] + sum;
	}
}		

__global__ void rank_gpu_kernel_7(int* partial_verify_vals,
		int* key_buff_ptr,
		int* test_rank_array,
		int* passed_verification_device,
		int iteration,
		int number_of_blocks,
		int amount_of_work){
	/*
	 * --------------------------------------------------------------------
	 * this is the partial verify test section 
	 * observe that test_rank_array vals are
	 * shifted differently for different cases
	 * --------------------------------------------------------------------
	 */
	int i, k;
	int passed_verification = 0;
	for(i=0; i<TEST_ARRAY_SIZE; i++){  
		/* test vals were put here on partial_verify_vals */                                           
		k = partial_verify_vals[i];          
		if(0<k && k<=NUM_KEYS-1){
			int key_rank = key_buff_ptr[k-1];
			int failed = 0;
			switch(CLASS){
				case 'S':
					if(i<=2){
						if(key_rank != test_rank_array[i]+iteration)
							failed = 1;
						else
							passed_verification++;
					}else{
						if(key_rank != test_rank_array[i]-iteration)
							failed = 1;
						else
							passed_verification++;
					}
					break;
				case 'W':
					if(i<2){
						if(key_rank != test_rank_array[i]+(iteration-2))
							failed = 1;
						else
							passed_verification++;
					}else{
						if(key_rank != test_rank_array[i]-iteration)
							failed = 1;
						else
							passed_verification++;
					}
					break;
				case 'A':
					if(i<=2){
						if(key_rank != test_rank_array[i]+(iteration-1))
							failed = 1;
						else
							passed_verification++;
					}else{
						if(key_rank != test_rank_array[i]-(iteration-1))
							failed = 1;
						else
							passed_verification++;
					}
					break;
				case 'B':
					if(i==1 || i==2 || i==4){
						if(key_rank != test_rank_array[i]+iteration)
							failed = 1;
						else
							passed_verification++;
					}
					else{
						if(key_rank != test_rank_array[i]-iteration)
							failed = 1;
						else
							passed_verification++;
					}
					break;
				case 'C':
					if(i<=2){
						if(key_rank != test_rank_array[i]+iteration)
							failed = 1;
						else
							passed_verification++;
					}else{
						if(key_rank != test_rank_array[i]-iteration)
							failed = 1;
						else
							passed_verification++;
					}
					break;
				case 'D':
					if(i<2){
						if(key_rank != test_rank_array[i]+iteration)
							failed = 1;
						else
							passed_verification++;
					}else{
						if(key_rank != test_rank_array[i]-iteration)
							failed = 1;
						else
							passed_verification++;
					}
					break;
			}
			if(failed==1){
				printf("Failed partial verification: iteration %d, test key %d\n", iteration, (int)i);
			}
		}
	}
	*passed_verification_device += passed_verification;
}

static void release_gpu(){
	cudaFree(key_array_device);
	cudaFree(key_buff1_device);
	cudaFree(key_buff2_device);
	cudaFree(index_array_device);
	cudaFree(rank_array_device);
	cudaFree(partial_verify_vals_device);
	cudaFree(passed_verification_device);
	cudaFree(key_scan_device);
	cudaFree(sum_device);
}

static void setup_gpu(){
	/*
	 * struct cudaDeviceProp{
	 *  char name[256];
	 *  size_t totalGlobalMem;
	 *  size_t sharedMemPerBlock;
	 *  int regsPerBlock;
	 *  int warpSize;
	 *  size_t memPitch;
	 *  int maxThreadsPerBlock;
	 *  int maxThreadsDim[3];
	 *  int maxGridSize[3];
	 *  size_t totalConstMem;
	 *  int major;
	 *  int minor;
	 *  int clockRate;
	 *  size_t textureAlignment;
	 *  int deviceOverlap;
	 *  int multiProcessorCount;
	 *  int kernelExecTimeoutEnabled;
	 *  int integrated;
	 *  int canMapHostMemory;
	 *  int computeMode;
	 *  int concurrentKernels;
	 *  int ECCEnabled;
	 *  int pciBusID;
	 *  int pciDeviceID;
	 *  int tccDriver;
	 * }
	 */
	/* amount of available devices */ 
	cudaGetDeviceCount(&total_devices);

	/* define gpu_device */
	if(total_devices==0){
		printf("\n\n\nNo Nvidia GPU found!\n\n\n");
		exit(-1);
	}else if((GPU_DEVICE>=0)&&
			(GPU_DEVICE<total_devices)){
		gpu_device_id = GPU_DEVICE;
	}else{
		gpu_device_id = 0;
	}
	cudaSetDevice(gpu_device_id);	
	cudaGetDeviceProperties(&gpu_device_properties, gpu_device_id);

	/* define threads_per_block */
	if((IS_THREADS_PER_BLOCK_ON_CREATE_SEQ>=1)&&
			(IS_THREADS_PER_BLOCK_ON_CREATE_SEQ<=gpu_device_properties.maxThreadsPerBlock)){
		threads_per_block_on_create_seq = IS_THREADS_PER_BLOCK_ON_CREATE_SEQ;
	}else{
		threads_per_block_on_create_seq = gpu_device_properties.warpSize;
	}
	if((IS_THREADS_PER_BLOCK_ON_RANK>=1)&&
			(IS_THREADS_PER_BLOCK_ON_RANK<=gpu_device_properties.maxThreadsPerBlock)){
		threads_per_block_on_rank = IS_THREADS_PER_BLOCK_ON_RANK;
	}else{
		threads_per_block_on_rank = gpu_device_properties.warpSize;
	}
	if((IS_THREADS_PER_BLOCK_ON_FULL_VERIFY>=1)&&
			(IS_THREADS_PER_BLOCK_ON_FULL_VERIFY<=gpu_device_properties.maxThreadsPerBlock)){
		threads_per_block_on_full_verify = IS_THREADS_PER_BLOCK_ON_FULL_VERIFY;
	}else{
		threads_per_block_on_full_verify = gpu_device_properties.warpSize;
	}	

	threads_per_block_on_rank_1=1;
	threads_per_block_on_rank_2=threads_per_block_on_rank;
	threads_per_block_on_rank_3=threads_per_block_on_rank;
	threads_per_block_on_rank_4=threads_per_block_on_rank;
	threads_per_block_on_rank_5=threads_per_block_on_rank;
	threads_per_block_on_rank_6=threads_per_block_on_rank;
	threads_per_block_on_rank_7=1;
	threads_per_block_on_full_verify_1=threads_per_block_on_full_verify;
	threads_per_block_on_full_verify_2=threads_per_block_on_full_verify;
	threads_per_block_on_full_verify_3=threads_per_block_on_full_verify;

	amount_of_work_on_create_seq=threads_per_block_on_create_seq*threads_per_block_on_create_seq;
	amount_of_work_on_rank_1=1;
	amount_of_work_on_rank_2=MAX_KEY;
	amount_of_work_on_rank_3=NUM_KEYS;
	amount_of_work_on_rank_4=threads_per_block_on_rank_4*threads_per_block_on_rank_4;
	amount_of_work_on_rank_5=threads_per_block_on_rank_5;
	amount_of_work_on_rank_6=threads_per_block_on_rank_6*threads_per_block_on_rank_6;
	amount_of_work_on_rank_7=1;
	amount_of_work_on_full_verify_1=NUM_KEYS;
	amount_of_work_on_full_verify_2=NUM_KEYS;
	amount_of_work_on_full_verify_3=NUM_KEYS;

	blocks_per_grid_on_create_seq=(ceil((double)(amount_of_work_on_create_seq)/(double)(threads_per_block_on_create_seq)));
	blocks_per_grid_on_rank_1=1;
	blocks_per_grid_on_rank_2=(ceil((double)(amount_of_work_on_rank_2)/(double)(threads_per_block_on_rank_2)));
	blocks_per_grid_on_rank_3=(ceil((double)(amount_of_work_on_rank_3)/(double)(threads_per_block_on_rank_3)));
	if(amount_of_work_on_rank_4 > MAX_KEY){amount_of_work_on_rank_4=MAX_KEY;}
	blocks_per_grid_on_rank_4=(ceil((double)(amount_of_work_on_rank_4)/(double)(threads_per_block_on_rank_4)));
	blocks_per_grid_on_rank_5=1;
	if(amount_of_work_on_rank_6 > MAX_KEY){amount_of_work_on_rank_6=MAX_KEY;}
	blocks_per_grid_on_rank_6=(ceil((double)(amount_of_work_on_rank_6)/(double)(threads_per_block_on_rank_6)));
	blocks_per_grid_on_rank_7=1;
	blocks_per_grid_on_full_verify_1=(ceil((double)(amount_of_work_on_full_verify_1)/(double)(threads_per_block_on_full_verify_1)));
	blocks_per_grid_on_full_verify_2=(ceil((double)(amount_of_work_on_full_verify_2)/(double)(threads_per_block_on_full_verify_2)));
	blocks_per_grid_on_full_verify_3=(ceil((double)(amount_of_work_on_full_verify_3)/(double)(threads_per_block_on_full_verify_3)));

	size_test_array_device=TEST_ARRAY_SIZE*sizeof(int);
	size_key_array_device=SIZE_OF_BUFFERS*sizeof(int); 
	size_key_buff1_device=MAX_KEY*sizeof(int); 
	size_key_buff2_device=SIZE_OF_BUFFERS*sizeof(int);
	size_index_array_device=TEST_ARRAY_SIZE*sizeof(int); 
	size_rank_array_device=TEST_ARRAY_SIZE*sizeof(int);
	size_partial_verify_vals_device=TEST_ARRAY_SIZE*sizeof(int);
	size_passed_verification_device=1*sizeof(int);
	size_key_scan_device=MAX_KEY*sizeof(int); 
	size_sum_device=threads_per_block_on_rank*sizeof(int);
	size_shared_data_on_rank_4=2*threads_per_block_on_rank_4*sizeof(int);
	size_shared_data_on_rank_5=2*threads_per_block_on_rank_5*sizeof(int);
	size_shared_data_on_full_verify_3=threads_per_block_on_full_verify_3*sizeof(int);

	cudaMalloc(&key_array_device, size_key_array_device);
	cudaMalloc(&key_buff1_device, size_key_buff1_device);
	cudaMalloc(&key_buff2_device, size_key_buff2_device);
	cudaMalloc(&index_array_device, size_index_array_device);
	cudaMalloc(&rank_array_device, size_rank_array_device);
	cudaMalloc(&partial_verify_vals_device, size_partial_verify_vals_device);
	cudaMalloc(&passed_verification_device, size_passed_verification_device);
	cudaMalloc(&key_scan_device, size_key_scan_device);
	cudaMalloc(&sum_device, size_sum_device);

	cudaMemcpy(index_array_device, test_index_array, size_index_array_device, cudaMemcpyHostToDevice);
	cudaMemcpy(rank_array_device, test_rank_array, size_rank_array_device, cudaMemcpyHostToDevice);
}
