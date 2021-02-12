/**
 * NASA Advanced Supercomputing Parallel Benchmarks C++
 *
 * based on NPB 3.3.1
 *
 * original version and technical report:
 * http://www.nas.nasa.gov/Software/NPB/
 *
 * Authors:
 *     M. Yarrow
 *     H. Jin
 *
 * C++ version:
 *     Dalvan Griebler <dalvangriebler@gmail.com>     
 *     Júnior Löff <loffjh@gmail.com>
 *     Gabriell Araujo <hexenoften@gmail.com>
 *
 * CUDA version:
 *     Gabriell Araujo <hexenoften@gmail.com>
 */

#include <cuda.h>
#include "../common/npb-CPP.hpp"
#include "npbparams.hpp"

#define PROFILING_TOTAL_TIME (0)
#define PROFILING_CREATE (1)
#define PROFILING_RANK (2)
#define PROFILING_VERIFY (3)

#define THREADS_PER_BLOCK (256)
#define SHARE_MEMORY_ON_RANK_GPU_KERNEL_4 (2*THREADS_PER_BLOCK)
#define SHARE_MEMORY_ON_RANK_GPU_KERNEL_5 (2*THREADS_PER_BLOCK)

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
int threads_per_block;
int blocks_per_grid;
int amount_of_work;
int THREADS_PER_BLOCK_AT_CREATE_SEQ_GPU_KERNEL;
int AMOUNT_OF_WORK_AT_CREATE_SEQ_GPU_KERNEL;
int THREADS_PER_BLOCK_AT_RANK_GPU_KERNEL_2;
int AMOUNT_OF_WORK_AT_RANK_GPU_KERNEL_2;
int THREADS_PER_BLOCK_AT_RANK_GPU_KERNEL_3;
int AMOUNT_OF_WORK_AT_RANK_GPU_KERNEL_3;
int THREADS_PER_BLOCK_AT_FULL_VERIFY;
int AMOUNT_OF_WORK_AT_FULL_VERIFY;
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
	int i, iteration;
	double timecounter;
	FILE* fp;

#if defined(PROFILING)
	printf(" PROFILING mode on\n");
#endif

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
	cudaMemcpy(&passed_verification, passed_verification_device, size_passed_verification_device, cudaMemcpyDeviceToHost);
#if defined(PROFILING)
	timer_stop(PROFILING_RANK);
#else
	timer_stop(PROFILING_TOTAL_TIME);
#endif

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

	/* end of timing, obtain maximum time of all processors */
#if defined(PROFILING)
	timer_stop(PROFILING_TOTAL_TIME);
	timecounter = timer_read(PROFILING_RANK);
#else
	timecounter = timer_read(PROFILING_TOTAL_TIME);
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
			(char*)CS1,
			(char*)CS2,
			(char*)CS3,
			(char*)CS4,
			(char*)CS5,
			(char*)CS6,
			(char*)CS7);

	/* print additional timers */
#if defined(PROFILING)
	double t_total, t_percent;
	t_total = timer_read(PROFILING_TOTAL_TIME);
	printf("\nAdditional timers -\n");
	printf(" Total execution: %8.3f\n", t_total);
	if(t_total == 0.0)t_total = 1.0;
	timecounter = timer_read(PROFILING_CREATE);
	t_percent = timecounter/t_total * 100.;
	printf(" Initialization : %8.3f (%5.2f%%)\n", timecounter, t_percent);
	timecounter = timer_read(PROFILING_RANK);
	t_percent = timecounter/t_total * 100.;
	printf(" Benchmarking   : %8.3f (%5.2f%%)\n", timecounter, t_percent);
	timecounter = timer_read(PROFILING_VERIFY);
	t_percent = timecounter/t_total * 100.;
	printf(" Sorting        : %8.3f (%5.2f%%)\n", timecounter, t_percent);
#endif

	release_gpu();

	return 0;  
}

static void create_seq_gpu(double seed, double a){  
	threads_per_block = THREADS_PER_BLOCK_AT_CREATE_SEQ_GPU_KERNEL;
	amount_of_work = AMOUNT_OF_WORK_AT_CREATE_SEQ_GPU_KERNEL;
	blocks_per_grid = (ceil((double)(amount_of_work)/(double)(threads_per_block)));

	create_seq_gpu_kernel<<<blocks_per_grid, threads_per_block>>>(key_array_device,
			seed,
			a,
			blocks_per_grid,
			amount_of_work);
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
	size_t size_memory_aux=sizeof(int)*(AMOUNT_OF_WORK_AT_FULL_VERIFY/THREADS_PER_BLOCK_AT_FULL_VERIFY);		
	cudaMalloc(&memory_aux_device, size_memory_aux);	

	/* full_verify_gpu_kernel_1 */
	threads_per_block = THREADS_PER_BLOCK;
	amount_of_work = NUM_KEYS;
	blocks_per_grid = (ceil((double)(amount_of_work)/(double)(threads_per_block)));
	full_verify_gpu_kernel_1<<<blocks_per_grid, threads_per_block>>>(key_array_device,
			key_buff2_device,
			blocks_per_grid,
			amount_of_work);
	cudaDeviceSynchronize();

	/* full_verify_gpu_kernel_2 */
	threads_per_block = THREADS_PER_BLOCK;
	amount_of_work = NUM_KEYS;
	blocks_per_grid = (ceil((double)(amount_of_work)/(double)(threads_per_block)));
	full_verify_gpu_kernel_2<<<blocks_per_grid, threads_per_block>>>(key_buff2_device,
			key_buff1_device,
			key_array_device,
			blocks_per_grid,
			amount_of_work);
	cudaDeviceSynchronize();

	/* full_verify_gpu_kernel_3 */
	threads_per_block = THREADS_PER_BLOCK_AT_FULL_VERIFY;
	amount_of_work = AMOUNT_OF_WORK_AT_FULL_VERIFY;
	blocks_per_grid = (ceil((double)(amount_of_work)/(double)(threads_per_block)));
	full_verify_gpu_kernel_3<<<blocks_per_grid, threads_per_block>>>(key_array_device,
			memory_aux_device,
			blocks_per_grid,
			amount_of_work);
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
	__shared__ int shared_aux[THREADS_PER_BLOCK];

	int i = (blockIdx.x*blockDim.x+threadIdx.x) + 1;

	if(i<NUM_KEYS){
		if(key_array[i-1]>key_array[i]){shared_aux[threadIdx.x]=1;}
		else{shared_aux[threadIdx.x]=0;}
	}else{shared_aux[threadIdx.x]=0;}

	__syncthreads();

	for(i=THREADS_PER_BLOCK/2; i>0; i>>=1){
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
	threads_per_block = 1;
	amount_of_work = 1;
	blocks_per_grid = 1;
	rank_gpu_kernel_1<<<blocks_per_grid, threads_per_block>>>(key_array_device,
			partial_verify_vals_device,
			index_array_device,
			iteration,
			blocks_per_grid,
			amount_of_work);
	cudaDeviceSynchronize();

	/* rank_gpu_kernel_2 */
	threads_per_block = THREADS_PER_BLOCK_AT_RANK_GPU_KERNEL_2;
	amount_of_work = AMOUNT_OF_WORK_AT_RANK_GPU_KERNEL_2;
	blocks_per_grid = (ceil((double)(amount_of_work)/(double)(threads_per_block)));
	rank_gpu_kernel_2<<<blocks_per_grid, threads_per_block>>>(key_buff1_device,
			blocks_per_grid,
			amount_of_work);
	cudaDeviceSynchronize();

	/* rank_gpu_kernel_3 */
	threads_per_block = THREADS_PER_BLOCK_AT_RANK_GPU_KERNEL_3;
	amount_of_work = AMOUNT_OF_WORK_AT_RANK_GPU_KERNEL_3;
	blocks_per_grid = (ceil((double)(amount_of_work)/(double)(threads_per_block)));
	rank_gpu_kernel_3<<<blocks_per_grid, threads_per_block>>>(key_buff1_device,
			key_array_device,
			blocks_per_grid,
			amount_of_work);
	cudaDeviceSynchronize();

	/* rank_gpu_kernel_4 */
	threads_per_block = THREADS_PER_BLOCK;
	amount_of_work = THREADS_PER_BLOCK * THREADS_PER_BLOCK;
	if(amount_of_work > MAX_KEY){amount_of_work = MAX_KEY;}
	blocks_per_grid = (ceil((double)(amount_of_work)/(double)(threads_per_block)));
	rank_gpu_kernel_4<<<blocks_per_grid, threads_per_block>>>(key_buff1_device,
			key_buff1_device,
			sum_device,
			blocks_per_grid,
			amount_of_work);
	cudaDeviceSynchronize();

	/* rank_gpu_kernel_5 */
	threads_per_block = THREADS_PER_BLOCK;
	amount_of_work = THREADS_PER_BLOCK;
	blocks_per_grid = (ceil((double)(amount_of_work)/(double)(threads_per_block)));
	rank_gpu_kernel_5<<<blocks_per_grid, threads_per_block>>>(sum_device,
			sum_device,
			blocks_per_grid,
			amount_of_work);
	cudaDeviceSynchronize();

	/* rank_gpu_kernel_6 */
	threads_per_block = THREADS_PER_BLOCK;
	amount_of_work = THREADS_PER_BLOCK * THREADS_PER_BLOCK;
	if(amount_of_work > MAX_KEY){amount_of_work = MAX_KEY;}
	blocks_per_grid = (ceil((double)(amount_of_work)/(double)(threads_per_block)));
	rank_gpu_kernel_6<<<blocks_per_grid, threads_per_block>>>(key_buff1_device,
			key_buff1_device,
			sum_device,
			blocks_per_grid,
			amount_of_work);
	cudaDeviceSynchronize();

	/* rank_gpu_kernel_7 */
	threads_per_block = 1;
	amount_of_work = 1;
	blocks_per_grid = 1;
	rank_gpu_kernel_7<<<blocks_per_grid, threads_per_block>>>(partial_verify_vals_device,
			key_buff1_device,
			rank_array_device,
			passed_verification_device,
			iteration,
			blocks_per_grid,
			amount_of_work);
	cudaDeviceSynchronize();
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
	int key = key_buff_ptr2[blockIdx.x*blockDim.x+threadIdx.x];
	atomicAdd(&key_buff_ptr[key], 1);
}

__global__ void rank_gpu_kernel_4(int* source,
		int* destiny,
		int* sum,
		int number_of_blocks,
		int amount_of_work){
	__shared__ int shared_data[SHARE_MEMORY_ON_RANK_GPU_KERNEL_4];

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
	__shared__ int shared_data[SHARE_MEMORY_ON_RANK_GPU_KERNEL_5];

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
	THREADS_PER_BLOCK_AT_CREATE_SEQ_GPU_KERNEL = 64;
	AMOUNT_OF_WORK_AT_CREATE_SEQ_GPU_KERNEL = THREADS_PER_BLOCK_AT_CREATE_SEQ_GPU_KERNEL * 256;
	THREADS_PER_BLOCK_AT_RANK_GPU_KERNEL_2 = THREADS_PER_BLOCK;
	AMOUNT_OF_WORK_AT_RANK_GPU_KERNEL_2 = MAX_KEY;
	THREADS_PER_BLOCK_AT_RANK_GPU_KERNEL_3 = THREADS_PER_BLOCK;
	AMOUNT_OF_WORK_AT_RANK_GPU_KERNEL_3 = NUM_KEYS;
	THREADS_PER_BLOCK_AT_FULL_VERIFY = THREADS_PER_BLOCK;
	AMOUNT_OF_WORK_AT_FULL_VERIFY = NUM_KEYS;

	size_test_array_device = sizeof(int) * TEST_ARRAY_SIZE;
	size_key_array_device = sizeof(int) * SIZE_OF_BUFFERS; 
	size_key_buff1_device = sizeof(int) * MAX_KEY; 
	size_key_buff2_device = sizeof(int) * SIZE_OF_BUFFERS;
	size_index_array_device = sizeof(int) * TEST_ARRAY_SIZE; 
	size_rank_array_device = sizeof(int) * TEST_ARRAY_SIZE;
	size_partial_verify_vals_device = sizeof(int) * TEST_ARRAY_SIZE;
	size_passed_verification_device = sizeof(int) * 1;
	size_key_scan_device = sizeof(int) * MAX_KEY; 
	size_sum_device = sizeof(int) * THREADS_PER_BLOCK;

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
