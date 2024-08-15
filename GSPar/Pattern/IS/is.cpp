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
 * The GSParLib version is a parallel implementation of the serial C++ version
 * GSParLib version: https://github.com/GMAP/NPB-GPU/tree/master/GSParLib
 *
 * Authors of the GSParLib code:
 *      Gabriell Araujo <hexenoften@gmail.com>
 *
 * ------------------------------------------------------------------------------
 * 
 * How to run:
 *      export LD_LIBRARY_PATH=../lib/gspar/bin:$LD_LIBRARY_PATH
 *      clear && make clean && make is CLASS=S GPU_DRIVER=CUDA && bin/is.S 
 *      clear && make clean && make is CLASS=S GPU_DRIVER=OPENCL && bin/is.S 
 * 
 * ------------------------------------------------------------------------------
 */

#include <iostream>
#include <chrono>

#include "../common/npb.hpp"
#include "npbparams.hpp"

#include "GSPar_PatternMap.hpp"
#include "GSPar_PatternReduce.hpp"
#include "GSPar_PatternComposition.hpp"

#ifdef GSPARDRIVER_CUDA
#include "GSPar_CUDA.hpp"
using namespace GSPar::Driver::CUDA;
#else
/* GSPARDRIVER_OPENCL */
#include "GSPar_OpenCL.hpp"
using namespace GSPar::Driver::OpenCL;
#endif

using namespace GSPar::Pattern;
using namespace std;

#define PROFILING_TOTAL_TIME (0)
#define IS_THREADS_PER_BLOCK_ON_RANK 256

/*****************************************************************/
/* For serial IS, buckets are not really req'd to solve NPB1 IS  */
/* spec, but their use on some machines improves performance, on */
/* other machines the use of buckets compromises performance,    */
/* probably because it is extra computation which is not req'd.  */
/* (Note: Mechanism not understood, probably cache related)      */
/* Example:  SP2-66MhzWN:  50% speedup with buckets              */
/* Example:  SGI Indy5000: 50% slowdown with buckets             */
/* Example:  SGI O2000:   400% slowdown with buckets (Wow!)      */
/*****************************************************************/
/* To disable the use of buckets, comment out the following line */
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
#define TOTAL_KEYS_LOG_2 16
#define MAX_KEY_LOG_2 11
#define NUM_BUCKETS_LOG_2 9
#endif

/*************/
/*  CLASS W  */
/*************/
#if CLASS == 'W'
#define TOTAL_KEYS_LOG_2 20
#define MAX_KEY_LOG_2 16
#define NUM_BUCKETS_LOG_2 10
#endif

/*************/
/*  CLASS A  */
/*************/
#if CLASS == 'A'
#define TOTAL_KEYS_LOG_2 23
#define MAX_KEY_LOG_2 19
#define NUM_BUCKETS_LOG_2 10
#endif

/*************/
/*  CLASS B  */
/*************/
#if CLASS == 'B'
#define TOTAL_KEYS_LOG_2 25
#define MAX_KEY_LOG_2 21
#define NUM_BUCKETS_LOG_2 10
#endif

/*************/
/*  CLASS C  */
/*************/
#if CLASS == 'C'
#define TOTAL_KEYS_LOG_2 27
#define MAX_KEY_LOG_2 23
#define NUM_BUCKETS_LOG_2 10
#endif

/*************/
/*  CLASS D  */
/*************/
#if CLASS == 'D'
#define TOTAL_KEYS_LOG_2 31
#define MAX_KEY_LOG_2 27
#define NUM_BUCKETS_LOG_2 10
#endif

#if CLASS == 'D'
#define TOTAL_KEYS (1L << TOTAL_KEYS_LOG_2)
#else
#define TOTAL_KEYS (1 << TOTAL_KEYS_LOG_2)
#endif
#define MAX_KEY (1 << MAX_KEY_LOG_2)
#define NUM_BUCKETS (1 << NUM_BUCKETS_LOG_2)
#define NUM_KEYS TOTAL_KEYS
#define SIZE_OF_BUFFERS NUM_KEYS                                           

#define MAX_ITERATIONS 10
#define TEST_ARRAY_SIZE 5

/*************************************/
/* Typedef: if necessary, change the */
/* size of int here by changing the  */
/* int type to, say, long            */
/*************************************/
#if CLASS == 'D'
typedef long INT_TYPE;
#else
typedef int INT_TYPE;
#endif

/********************/
/* Some global info */
/********************/
INT_TYPE* key_buff_ptr_global; /* used by full_verify to get copies of rank info */
int passed_verification;                                 

/************************************/
/* These are the three main arrays. */
/* See SIZE_OF_BUFFERS def above    */
/************************************/
#if defined(DO_NOT_ALLOCATE_ARRAYS_WITH_DYNAMIC_MEMORY_AND_AS_SINGLE_DIMENSION)
INT_TYPE key_array[SIZE_OF_BUFFERS];
INT_TYPE key_buff1[MAX_KEY];
INT_TYPE key_buff2[SIZE_OF_BUFFERS];
INT_TYPE partial_verify_vals[TEST_ARRAY_SIZE];
INT_TYPE** key_buff1_aptr = NULL;
#else
INT_TYPE (*key_array)=(INT_TYPE*)malloc(sizeof(INT_TYPE)*(SIZE_OF_BUFFERS));    
INT_TYPE (*key_buff1)=(INT_TYPE*)malloc(sizeof(INT_TYPE)*(MAX_KEY));                
INT_TYPE (*key_buff2)=(INT_TYPE*)malloc(sizeof(INT_TYPE)*(SIZE_OF_BUFFERS));                
INT_TYPE (*partial_verify_vals)=(INT_TYPE*)malloc(sizeof(INT_TYPE)*(TEST_ARRAY_SIZE));       
INT_TYPE** key_buff1_aptr = NULL;
#endif

#ifdef USE_BUCKETS
#if defined(DO_NOT_ALLOCATE_ARRAYS_WITH_DYNAMIC_MEMORY_AND_AS_SINGLE_DIMENSION)
INT_TYPE** bucket_size; 
INT_TYPE bucket_ptrs[NUM_BUCKETS];
#else
INT_TYPE** bucket_size; 
INT_TYPE (*bucket_ptrs)=(INT_TYPE*)malloc(sizeof(INT_TYPE)*(NUM_BUCKETS));
#endif
#endif

/**********************/
/* Partial verif info */
/**********************/
INT_TYPE test_index_array[TEST_ARRAY_SIZE],
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

/***********************/
/* function prototypes */
/***********************/
void alloc_key_buff();
void* alloc_mem(size_t size);
void create_seq(double seed, 
		double a);
double find_my_seed(int kn,
		int np,
		long nn,
		double s,
		double a );
void full_verify();
void rank_host(int iteration);
void rank_gpu(int iteration);
void setup_gpu();

/* gpu data */
static int sum_device[IS_THREADS_PER_BLOCK_ON_RANK*IS_THREADS_PER_BLOCK_ON_RANK];
static int summ_device[IS_THREADS_PER_BLOCK_ON_RANK*IS_THREADS_PER_BLOCK_ON_RANK];
//
extern std::string source_additional_routines;	
//
string DEVICE_NAME;
Instance* driver;
//
MemoryObject* test_index_array_device;
MemoryObject* test_rank_array_device;
MemoryObject* key_array_device;
MemoryObject* partial_verify_vals_device;
MemoryObject* key_buff1_device;
MemoryObject* summ_device_device;
MemoryObject* passed_verification_device;
//
size_t size_test_index_array_device;
size_t size_test_rank_array_device;
size_t size_key_array_device;
size_t size_partial_verify_vals_device;
size_t size_key_buff1_device;
size_t size_summ_device_device;
size_t size_passed_verification_device;
//
int threads_per_block_on_rank;
int threads_per_block_on_rank_1;
int threads_per_block_on_rank_2;
int threads_per_block_on_rank_3;
int threads_per_block_on_rank_4;
int threads_per_block_on_rank_5;
int threads_per_block_on_rank_6;
int threads_per_block_on_rank_7;
//
int blocks_per_grid_on_rank_1;
int blocks_per_grid_on_rank_2;
int blocks_per_grid_on_rank_3;
int blocks_per_grid_on_rank_4;
int blocks_per_grid_on_rank_5;
int blocks_per_grid_on_rank_6;
int blocks_per_grid_on_rank_7;
//
int amount_of_work_on_rank_1;
int amount_of_work_on_rank_2;
int amount_of_work_on_rank_3;
int amount_of_work_on_rank_4;
int amount_of_work_on_rank_5;
int amount_of_work_on_rank_6;
int amount_of_work_on_rank_7;
//
Map* kernel_rank_1;
Map* kernel_rank_2;
Map* kernel_rank_3;
Map* kernel_rank_4;
Map* kernel_rank_5;
Map* kernel_rank_6;
Map* kernel_rank_7;
//
extern std::string source_kernel_rank_1;
extern std::string source_kernel_rank_2;
extern std::string source_kernel_rank_3;
extern std::string source_kernel_rank_4;
extern std::string source_kernel_rank_5;
extern std::string source_kernel_rank_6;
extern std::string source_kernel_rank_7;

/*****************************************************************/
/*************             M  A  I  N             ****************/
/*****************************************************************/
int main(int argc, char** argv){
#if defined(DO_NOT_ALLOCATE_ARRAYS_WITH_DYNAMIC_MEMORY_AND_AS_SINGLE_DIMENSION)
	printf(" DO_NOT_ALLOCATE_ARRAYS_WITH_DYNAMIC_MEMORY_AND_AS_SINGLE_DIMENSION mode on\n");
#endif
#ifdef GSPARDRIVER_CUDA
	printf(" Performing GSParLib with CUDA\n");
#else
/* GSPARDRIVER_OPENCL */
	printf(" Performing GSParLib with OpenCL\n");
#endif
	int i, iteration;
	double timecounter;
	FILE* fp;

	/* Initialize the verification arrays if a valid class */
	for( i=0; i<TEST_ARRAY_SIZE; i++ )
		switch( CLASS )
		{
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

	/* Printout initial NPB info */
	printf("\n\n NAS Parallel Benchmarks 4.1 Serial C++ version - IS Benchmark\n\n");
	printf(" Size:  %ld  (class %c)\n", (long)TOTAL_KEYS, CLASS);
	printf(" Iterations:   %d\n", MAX_ITERATIONS);
	printf( "\n" );

	/* Generate random number sequence and subsequent keys on all procs */
	create_seq(314159265.00 /* Random number gen seed */, 
			1220703125.00 /* Random number gen mult */); 

	alloc_key_buff();

	/* Do one interation for free (i.e., untimed) to guarantee initialization of */
	/* all data and code pages and respective tables */
	rank_host( 1 ); 

	/* Start verification counter */
	passed_verification = 0;

	if( CLASS != 'S' ) printf( "\n   iteration\n" );

	setup_gpu();

	timer_clear(PROFILING_TOTAL_TIME);          
	timer_start(PROFILING_TOTAL_TIME);

	/* This is the main iteration */
	for(iteration=1; iteration<=MAX_ITERATIONS; iteration++){
		if(CLASS != 'S')printf("        %d\n", iteration);
		rank_gpu( iteration );
	}

	/* End of timing, obtain maximum time of all processors */
	timer_stop(PROFILING_TOTAL_TIME);
	timecounter = timer_read(PROFILING_TOTAL_TIME);
	
	// copy results to host
	test_index_array_device->copyOut();
	test_rank_array_device->copyOut();
	key_array_device->copyOut();
	partial_verify_vals_device->copyOut();
	key_buff1_device->copyOut();
	summ_device_device->copyOut();
	passed_verification_device->copyOut();
	key_buff_ptr_global = key_buff1; // key_buff_ptr;

	/* This tests that keys are in sequence: sorting of last ranked key seq */
	/* occurs here, but is an untimed operation */
	full_verify();    

	/* The final printout */
	if(passed_verification != 5*MAX_ITERATIONS + 1){passed_verification = 0;}
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
			(char*) DEVICE_NAME.c_str(),
			(char*)CS1,
			(char*)CS2,
			(char*)CS3,
			(char*)CS4,
			(char*)CS5,
			(char*)CS6,
			(char*)CS7);

	return 0;
}

void alloc_key_buff(){
	INT_TYPE i;
	int num_procs;

	num_procs = 1;

#ifdef USE_BUCKETS
	bucket_size = (INT_TYPE**)alloc_mem(sizeof(INT_TYPE*)*num_procs);

	for(i = 0; i < num_procs; i++){
		bucket_size[i] = (INT_TYPE*)alloc_mem(sizeof(INT_TYPE)*NUM_BUCKETS);
	}

	for( i=0; i<NUM_KEYS; i++ )
		key_buff2[i] = 0;
#else /*USE_BUCKETS*/
	key_buff1_aptr = (INT_TYPE**)alloc_mem(sizeof(INT_TYPE*)*num_procs);

	key_buff1_aptr[0] = key_buff1;
	for(i = 1; i < num_procs; i++) {
		key_buff1_aptr[i] = (INT_TYPE *)alloc_mem(sizeof(INT_TYPE) * MAX_KEY);
	}
#endif /*USE_BUCKETS*/
}

/*****************************************************************/
/*****************    Allocate Working Buffer     ****************/
/*****************************************************************/
void* alloc_mem(size_t size){
	void* p;
	p = (void*)malloc(size);
	if(!p){
		perror("Memory allocation error");
		exit(1);
	}
	return p;
}

void create_seq(double seed, double a){
	double x;
	int    i, k;

	k = MAX_KEY/4;

	for(i=0; i<NUM_KEYS; i++){
		x = randlc(&seed, a);
		x += randlc(&seed, a);
		x += randlc(&seed, a);
		x += randlc(&seed, a);    

		key_array[i] = k*x;
	}
}

/*****************************************************************/
/************   F  I  N  D  _  M  Y  _  S  E  E  D    ************/
/************                                         ************/
/************ returns parallel random number seq seed ************/
/*****************************************************************/
double find_my_seed(int kn, /* my processor rank, 0<=kn<=num procs */
		int np, /* np = num procs */
		long nn, /* total num of ran numbers, all procs */
		double s, /* Ran num seed, for ex.: 314159265.00 */
		double a){ /* Ran num gen mult, try 1220703125.00 */
	/*
	 * Create a random number sequence of total length nn residing
	 * on np number of processors.  Each processor will therefore have a
	 * subsequence of length nn/np.  This routine returns that random
	 * number which is the first random number for the subsequence belonging
	 * to processor rank kn, and which is used as seed for proc kn ran # gen.
	 */
	double t1,t2;
	long mq,nq,kk,ik;

	if ( kn == 0 ) return s;

	mq = (nn/4 + np - 1) / np;
	nq = mq * 4 * kn; /* number of rans to be skipped */

	t1 = s;
	t2 = a;
	kk = nq;
	while( kk > 1 ){
		ik = kk / 2;
		if(2 * ik ==  kk){
			(void)randlc( &t2, t2 );
			kk = ik;
		}
		else{
			(void)randlc( &t1, t2 );
			kk = kk - 1;
		}
	}
	(void)randlc( &t1, t2 );

	return( t1 );
}

void full_verify(){
	INT_TYPE    i, j;

	/* now, finally, sort the keys: */
#ifdef USE_BUCKETS
	/* key_buff2[] already has the proper information, so do nothing */
#else
	/* copy keys into work array; keys in key_array will be reassigned. */
	for(i=0; i<NUM_KEYS; i++){key_buff2[i] = key_array[i];}
#endif
	for(i=0; i<NUM_KEYS; i++){key_buff2[i] = key_array[i];}

	for(i=0; i<NUM_KEYS; i++){key_array[--key_buff_ptr_global[key_buff2[i]]] = key_buff2[i];}

	/* confirm keys correctly sorted: count incorrectly sorted keys, if any */
	j = 0;
	for(i=1; i<NUM_KEYS; i++){if(key_array[i-1] > key_array[i]){j++;}}

	if(j != 0){printf( "Full_verify: number of keys out of sort: %ld\n",(long)j);}
	else{passed_verification++;}
}

/*****************************************************************/
/*************             R  A  N  K             ****************/
/*****************************************************************/
void rank_host(int iteration){
	INT_TYPE i, k;
	INT_TYPE *key_buff_ptr, *key_buff_ptr2;

#ifdef USE_BUCKETS
	int shift = MAX_KEY_LOG_2 - NUM_BUCKETS_LOG_2;
	INT_TYPE num_bucket_keys = (1L << shift);
#endif

	key_array[iteration] = iteration;
	key_array[iteration+MAX_ITERATIONS] = MAX_KEY - iteration;

	/* Determine where the partial verify test keys are, load into */
	/* top of array bucket_size */
	for( i=0; i<TEST_ARRAY_SIZE; i++ )
		partial_verify_vals[i] = key_array[test_index_array[i]];

	/* Setup pointers to key buffers */
#ifdef USE_BUCKETS
	key_buff_ptr2 = key_buff2;
#else
	key_buff_ptr2 = key_array;
#endif
	key_buff_ptr = key_buff1;

	INT_TYPE *work_buff, m, k1, k2;
	int myid = 0, num_procs = 1;

	/* Bucket sort is known to improve cache performance on some */
	/* cache based systems.  But the actual performance may depend */
	/* on cache size, problem size. */
#ifdef USE_BUCKETS
	work_buff = bucket_size[myid];

	/* Initialize */
	for( i=0; i<NUM_BUCKETS; i++ )  
		work_buff[i] = 0;

	/* Determine the number of keys in each bucket */
	for( i=0; i<NUM_KEYS; i++ )
		work_buff[key_array[i] >> shift]++;

	/* Accumulative bucket sizes are the bucket pointers. */
	/* These are global sizes accumulated upon to each bucket */
	bucket_ptrs[0] = 0;
	for( k=0; k< myid; k++ )  
		bucket_ptrs[0] += bucket_size[k][0];

	for( i=1; i< NUM_BUCKETS; i++ ) { 
		bucket_ptrs[i] = bucket_ptrs[i-1];
		for( k=0; k< myid; k++ )
			bucket_ptrs[i] += bucket_size[k][i];
		for( k=myid; k< num_procs; k++ )
			bucket_ptrs[i] += bucket_size[k][i-1];
	}

	/* Sort into appropriate bucket */
	for( i=0; i<NUM_KEYS; i++ ){
		k = key_array[i];
		key_buff2[bucket_ptrs[k >> shift]++] = k;
	}

	/* The bucket pointers now point to the final accumulated sizes */
	if (myid < num_procs-1) {
		for( i=0; i< NUM_BUCKETS; i++ )
			for( k=myid+1; k< num_procs; k++ )
				bucket_ptrs[i] += bucket_size[k][i];
	}

	/* Now, buckets are sorted.  We only need to sort keys inside */
	/* each bucket, which can be done in parallel.  Because the distribution */
	/* of the number of keys in the buckets is Gaussian, the use of */
	/* a dynamic schedule should improve load balance, thus, performance */
	for( i=0; i< NUM_BUCKETS; i++ ) {
		/* Clear the work array section associated with each bucket */
		k1 = i * num_bucket_keys;
		k2 = k1 + num_bucket_keys;
		for ( k = k1; k < k2; k++ )
			key_buff_ptr[k] = 0;
		/* Ranking of all keys occurs in this section: */
		/* In this section, the keys themselves are used as their */
		/* own indexes to determine how many of each there are: their */
		/* individual population */
		m = (i > 0)? bucket_ptrs[i-1] : 0;
		for ( k = m; k < bucket_ptrs[i]; k++ )
			key_buff_ptr[key_buff_ptr2[k]]++; /* Now they have individual key population */
		/* To obtain ranks of each key, successively add the individual key */
		/* population, not forgetting to add m, the total of lesser keys, */
		/* to the first key population */
		key_buff_ptr[k1] += m;
		for ( k = k1+1; k < k2; k++ )
			key_buff_ptr[k] += key_buff_ptr[k-1];
	}
#else /*USE_BUCKETS*/
	work_buff = key_buff1_aptr[myid];
	/* Clear the work array */
	for( i=0; i<MAX_KEY; i++ )
		work_buff[i] = 0;
	/* Ranking of all keys occurs in this section: */
	/* In this section, the keys themselves are used as their */
	/* own indexes to determine how many of each there are: their */
	/* individual population */
	for( i=0; i<NUM_KEYS; i++ )
		work_buff[key_buff_ptr2[i]]++; /* Now they have individual key population */
	/* To obtain ranks of each key, successively add the individual key population */
	for( i=0; i<MAX_KEY-1; i++ )   
		work_buff[i+1] += work_buff[i];
	/* Accumulate the global key population */
	for( k=1; k<num_procs; k++ ){
		for( i=0; i<MAX_KEY; i++ )
			key_buff_ptr[i] += key_buff1_aptr[k][i];
	}
#endif /*USE_BUCKETS*/  

	/* This is the partial verify test section */
	/* Observe that test_rank_array vals are */
	/* shifted differently for different cases */
	for( i=0; i<TEST_ARRAY_SIZE; i++ ){                                             
		k = partial_verify_vals[i]; /* test vals were put here */
		if( 0 < k  &&  k <= NUM_KEYS-1 )
		{
			INT_TYPE key_rank = key_buff_ptr[k-1];
			int failed = 0;

			switch( CLASS )
			{
				case 'S':
					if( i <= 2 )
					{
						if( key_rank != test_rank_array[i]+iteration )
							failed = 1;
						else
							passed_verification++;
					}
					else
					{
						if( key_rank != test_rank_array[i]-iteration )
							failed = 1;
						else
							passed_verification++;
					}
					break;
				case 'W':
					if( i < 2 )
					{
						if( key_rank != test_rank_array[i]+(iteration-2) )
							failed = 1;
						else
							passed_verification++;
					}
					else
					{
						if( key_rank != test_rank_array[i]-iteration )
							failed = 1;
						else
							passed_verification++;
					}
					break;
				case 'A':
					if( i <= 2 )
					{
						if( key_rank != test_rank_array[i]+(iteration-1) )
							failed = 1;
						else
							passed_verification++;
					}
					else
					{
						if( key_rank != test_rank_array[i]-(iteration-1) )
							failed = 1;
						else
							passed_verification++;
					}
					break;
				case 'B':
					if( i == 1 || i == 2 || i == 4 )
					{
						if( key_rank != test_rank_array[i]+iteration )
							failed = 1;
						else
							passed_verification++;
					}
					else
					{
						if( key_rank != test_rank_array[i]-iteration )
							failed = 1;
						else
							passed_verification++;
					}
					break;
				case 'C':
					if( i <= 2 )
					{
						if( key_rank != test_rank_array[i]+iteration )
							failed = 1;
						else
							passed_verification++;
					}
					else
					{
						if( key_rank != test_rank_array[i]-iteration )
							failed = 1;
						else
							passed_verification++;
					}
					break;
				case 'D':
					if( i < 2 )
					{
						if( key_rank != test_rank_array[i]+iteration )
							failed = 1;
						else
							passed_verification++;
					}
					else
					{
						if( key_rank != test_rank_array[i]-iteration )
							failed = 1;
						else
							passed_verification++;
					}
					break;
			}
			if( failed == 1 )
				printf( "Failed partial verification: "
						"iteration %d, test key %d\n", 
						iteration, (int)i );
		}
	}

	/* Make copies of rank info for use by full_verify: these variables */
	/* in rank are local; making them global slows down the code, probably */
	/* since they cannot be made register by compiler */
	if( iteration == MAX_ITERATIONS ) 
		key_buff_ptr_global = key_buff_ptr;
}

void rank_gpu(int iteration){
	/* kernel rank 1 */
	try {
		kernel_rank_1->setParameter<int*>("key_array", key_array_device, GSPAR_PARAM_PRESENT);
		kernel_rank_1->setParameter<int*>("partial_verify_vals", partial_verify_vals_device, GSPAR_PARAM_PRESENT);
		kernel_rank_1->setParameter<int*>("test_index_array", test_index_array_device, GSPAR_PARAM_PRESENT);
		kernel_rank_1->setParameter("iteration", iteration);
		kernel_rank_1->setParameter("MAX_ITERATIONS", MAX_ITERATIONS);
		kernel_rank_1->setParameter("MAX_KEY", MAX_KEY);
		kernel_rank_1->setParameter("TEST_ARRAY_SIZE", TEST_ARRAY_SIZE);	

		kernel_rank_1->run<Instance>();
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}

	/* kernel rank 2 */
	try {
		kernel_rank_2->setParameter<int*>("key_buff1", key_buff1_device, GSPAR_PARAM_PRESENT);
		kernel_rank_2->setParameter("MAX_KEY", MAX_KEY);

		kernel_rank_2->run<Instance>();
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}

	/* kernel rank 3 */
	try {
		kernel_rank_3->setParameter<int*>("key_buff1", key_buff1_device, GSPAR_PARAM_PRESENT);
		kernel_rank_3->setParameter<int*>("key_array", key_array_device, GSPAR_PARAM_PRESENT);

		kernel_rank_3->run<Instance>();
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}

	/* kernel rank 4 */
	try {
		kernel_rank_4->setParameter<int*>("source", key_buff1_device, GSPAR_PARAM_PRESENT);
		kernel_rank_4->setParameter<int*>("sum", summ_device_device, GSPAR_PARAM_PRESENT);
		kernel_rank_4->setParameter("MAX_KEY", MAX_KEY);
		kernel_rank_4->setParameter("number_of_blocks", blocks_per_grid_on_rank_4);

		kernel_rank_4->run<Instance>();
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}

	/* kernel rank 5 */
	try {
		kernel_rank_5->setParameter<int*>("source", summ_device_device, GSPAR_PARAM_PRESENT);
		kernel_rank_5->setParameter<int*>("destiny", summ_device_device, GSPAR_PARAM_PRESENT);
		kernel_rank_5->setParameter("MAX_KEY", MAX_KEY);

		kernel_rank_5->run<Instance>();
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}

	/* kernel rank 6 */
	try {
		kernel_rank_6->setParameter<int*>("source", key_buff1_device, GSPAR_PARAM_PRESENT);
		kernel_rank_6->setParameter<int*>("offset", summ_device_device, GSPAR_PARAM_PRESENT);
		kernel_rank_6->setParameter("MAX_KEY", MAX_KEY);
		kernel_rank_6->setParameter("number_of_blocks", blocks_per_grid_on_rank_6);	

		kernel_rank_6->run<Instance>();
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}

	/* kernel rank 7 */
	try {
		kernel_rank_7->setParameter<int*>("partial_verify_vals", partial_verify_vals_device, GSPAR_PARAM_PRESENT);
		kernel_rank_7->setParameter<int*>("key_buff_ptr", key_buff1_device, GSPAR_PARAM_PRESENT);
		kernel_rank_7->setParameter<int*>("test_rank_array", test_rank_array_device, GSPAR_PARAM_PRESENT);
		kernel_rank_7->setParameter<int*>("passed_verification_device", passed_verification_device, GSPAR_PARAM_PRESENT);
		kernel_rank_7->setParameter("iteration", iteration);
		kernel_rank_7->setParameter("TEST_ARRAY_SIZE", TEST_ARRAY_SIZE);
		kernel_rank_7->setParameter("CLASS", CLASS);
		kernel_rank_7->setParameter("NUM_KEYS", NUM_KEYS);

		kernel_rank_7->run<Instance>();
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}
}

void setup_gpu(){
	driver = Instance::getInstance();
	driver->init();

	int numGpus = driver->getGpuCount();
	if (numGpus == 0) {
		std::cout << "No GPU found, interrupting the benchmark" << std::endl;
		exit(-1);
	}

	auto gpus = driver->getGpuList();

	DEVICE_NAME = gpus[0]->getName();	

	auto gpu = driver->getGpu(0);

	threads_per_block_on_rank = IS_THREADS_PER_BLOCK_ON_RANK;

	threads_per_block_on_rank_1=1;
	threads_per_block_on_rank_2=threads_per_block_on_rank;
	threads_per_block_on_rank_3=threads_per_block_on_rank;
	threads_per_block_on_rank_4=threads_per_block_on_rank;
	threads_per_block_on_rank_5=threads_per_block_on_rank;
	threads_per_block_on_rank_6=threads_per_block_on_rank;
	threads_per_block_on_rank_7=1;

	amount_of_work_on_rank_1=1;
	amount_of_work_on_rank_2=MAX_KEY;
	amount_of_work_on_rank_3=NUM_KEYS;
	amount_of_work_on_rank_4=threads_per_block_on_rank_4*threads_per_block_on_rank_4;
	amount_of_work_on_rank_5=threads_per_block_on_rank_5;
	amount_of_work_on_rank_6=threads_per_block_on_rank_6*threads_per_block_on_rank_6;
	amount_of_work_on_rank_7=1;

	blocks_per_grid_on_rank_1=1;
	blocks_per_grid_on_rank_2=(ceil((double)(amount_of_work_on_rank_2)/(double)(threads_per_block_on_rank_2)));
	blocks_per_grid_on_rank_3=(ceil((double)(amount_of_work_on_rank_3)/(double)(threads_per_block_on_rank_3)));
	if(amount_of_work_on_rank_4 > MAX_KEY){amount_of_work_on_rank_4=MAX_KEY;}
	blocks_per_grid_on_rank_4=(ceil((double)(amount_of_work_on_rank_4)/(double)(threads_per_block_on_rank_4)));
	blocks_per_grid_on_rank_5=1;
	if(amount_of_work_on_rank_6 > MAX_KEY){amount_of_work_on_rank_6=MAX_KEY;}
	blocks_per_grid_on_rank_6=(ceil((double)(amount_of_work_on_rank_6)/(double)(threads_per_block_on_rank_6)));
	blocks_per_grid_on_rank_7=1;

	size_test_index_array_device = sizeof(int) * TEST_ARRAY_SIZE;
	size_test_rank_array_device = sizeof(int) * TEST_ARRAY_SIZE;
	size_key_array_device = sizeof(int) * SIZE_OF_BUFFERS;
	size_partial_verify_vals_device = sizeof(int) * TEST_ARRAY_SIZE;
	size_key_buff1_device = sizeof(int) * MAX_KEY;
	size_summ_device_device = sizeof(int) * IS_THREADS_PER_BLOCK_ON_RANK * IS_THREADS_PER_BLOCK_ON_RANK;
	size_passed_verification_device = sizeof(int);

	test_index_array_device = gpu->malloc(size_test_index_array_device, test_index_array);
	test_rank_array_device = gpu->malloc(size_test_rank_array_device, test_rank_array);
	key_array_device = gpu->malloc(size_key_array_device, key_array);
	partial_verify_vals_device = gpu->malloc(size_partial_verify_vals_device, partial_verify_vals);
	key_buff1_device = gpu->malloc(size_key_buff1_device, key_buff1);
	summ_device_device = gpu->malloc(size_summ_device_device, summ_device);
	passed_verification_device = gpu->malloc(size_passed_verification_device, &passed_verification);

	test_index_array_device->copyIn();
	test_rank_array_device->copyIn();
	key_array_device->copyIn();
	partial_verify_vals_device->copyIn();
	key_buff1_device->copyIn();
	summ_device_device->copyIn();
	passed_verification_device->copyIn();

	int iteration = 0;

	/* compiling each pattern */
	/* kernel rank 1 */
	try {
		unsigned long dims[3] = {(long unsigned int)amount_of_work_on_rank_1, 0, 0}; 
		kernel_rank_1 = new Map(source_kernel_rank_1);
		kernel_rank_1->setStdVarNames({"gspar_thread_id"});
		kernel_rank_1->setParameter<int*>("key_array", key_array_device, GSPAR_PARAM_PRESENT);
		kernel_rank_1->setParameter<int*>("partial_verify_vals", partial_verify_vals_device, GSPAR_PARAM_PRESENT);
		kernel_rank_1->setParameter<int*>("test_index_array", test_index_array_device, GSPAR_PARAM_PRESENT);
		kernel_rank_1->setParameter("iteration", iteration);
		kernel_rank_1->setParameter("MAX_ITERATIONS", MAX_ITERATIONS);
		kernel_rank_1->setParameter("MAX_KEY", MAX_KEY);
		kernel_rank_1->setParameter("TEST_ARRAY_SIZE", TEST_ARRAY_SIZE);	
		kernel_rank_1->setNumThreadsPerBlockForX(threads_per_block_on_rank_1);
		kernel_rank_1->addExtraKernelCode(source_additional_routines);
		kernel_rank_1->compile<Instance>(dims);
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}

	/* kernel rank 2 */
	try {
		unsigned long dims[3] = {(long unsigned int)amount_of_work_on_rank_2, 0, 0}; 
		kernel_rank_2 = new Map(source_kernel_rank_2);
		kernel_rank_2->setStdVarNames({"gspar_thread_id"});
		kernel_rank_2->setParameter<int*>("key_buff1", key_buff1_device, GSPAR_PARAM_PRESENT);
		kernel_rank_2->setParameter("MAX_KEY", MAX_KEY);
		kernel_rank_2->setNumThreadsPerBlockForX(threads_per_block_on_rank_2);
		kernel_rank_2->addExtraKernelCode(source_additional_routines);
		kernel_rank_2->compile<Instance>(dims);
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}

	/* kernel rank 3 */
	try {
		unsigned long dims[3] = {(long unsigned int)amount_of_work_on_rank_3, 0, 0}; 
		kernel_rank_3 = new Map(source_kernel_rank_3);
		kernel_rank_3->setStdVarNames({"gspar_thread_id"});
		kernel_rank_3->setParameter<int*>("key_buff1", key_buff1_device, GSPAR_PARAM_PRESENT);
		kernel_rank_3->setParameter<int*>("key_array", key_array_device, GSPAR_PARAM_PRESENT);
		kernel_rank_3->setNumThreadsPerBlockForX(threads_per_block_on_rank_3);
		kernel_rank_3->addExtraKernelCode(source_additional_routines);
		kernel_rank_3->compile<Instance>(dims);
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}

	/* kernel rank 4 */
	try {
		unsigned long dims[3] = {(long unsigned int)amount_of_work_on_rank_4, 0, 0}; 
		kernel_rank_4 = new Map(source_kernel_rank_4);
		kernel_rank_4->setStdVarNames({"gspar_thread_id"});
		kernel_rank_4->setParameter<int*>("source", key_buff1_device, GSPAR_PARAM_PRESENT);
		kernel_rank_4->setParameter<int*>("sum", summ_device_device, GSPAR_PARAM_PRESENT);
		kernel_rank_4->setParameter("MAX_KEY", MAX_KEY);
		kernel_rank_4->setParameter("number_of_blocks", blocks_per_grid_on_rank_4);
		kernel_rank_4->setNumThreadsPerBlockForX(threads_per_block_on_rank_4);
		kernel_rank_4->addExtraKernelCode(source_additional_routines);
		kernel_rank_4->compile<Instance>(dims);
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}

	/* kernel rank 5 */
	try {
		unsigned long dims[3] = {(long unsigned int)amount_of_work_on_rank_5, 0, 0}; 
		kernel_rank_5 = new Map(source_kernel_rank_5);
		kernel_rank_5->setStdVarNames({"gspar_thread_id"});
		kernel_rank_5->setParameter<int*>("source", summ_device_device, GSPAR_PARAM_PRESENT);
		kernel_rank_5->setParameter<int*>("destiny", summ_device_device, GSPAR_PARAM_PRESENT);
		kernel_rank_5->setParameter("MAX_KEY", MAX_KEY);	
		kernel_rank_5->setNumThreadsPerBlockForX(threads_per_block_on_rank_5);
		kernel_rank_5->addExtraKernelCode(source_additional_routines);
		kernel_rank_5->compile<Instance>(dims);
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}

	/* kernel rank 6 */
	try {
		unsigned long dims[3] = {(long unsigned int)amount_of_work_on_rank_6, 0, 0}; 
		kernel_rank_6 = new Map(source_kernel_rank_6);
		kernel_rank_6->setStdVarNames({"gspar_thread_id"});
		kernel_rank_6->setParameter<int*>("source", key_buff1_device, GSPAR_PARAM_PRESENT);
		kernel_rank_6->setParameter<int*>("offset", summ_device_device, GSPAR_PARAM_PRESENT);
		kernel_rank_6->setParameter("MAX_KEY", MAX_KEY);
		kernel_rank_6->setParameter("number_of_blocks", blocks_per_grid_on_rank_6);	
		kernel_rank_6->setNumThreadsPerBlockForX(threads_per_block_on_rank_6);
		kernel_rank_6->addExtraKernelCode(source_additional_routines);
		kernel_rank_6->compile<Instance>(dims);
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}

	/* kernel rank 7 */
	try {
		unsigned long dims[3] = {(long unsigned int)amount_of_work_on_rank_7, 0, 0}; 
		kernel_rank_7 = new Map(source_kernel_rank_7);
		kernel_rank_7->setStdVarNames({"gspar_thread_id"});
		kernel_rank_7->setParameter<int*>("partial_verify_vals", partial_verify_vals_device, GSPAR_PARAM_PRESENT);
		kernel_rank_7->setParameter<int*>("key_buff_ptr", key_buff1_device, GSPAR_PARAM_PRESENT);
		kernel_rank_7->setParameter<int*>("test_rank_array", test_rank_array_device, GSPAR_PARAM_PRESENT);
		kernel_rank_7->setParameter<int*>("passed_verification_device", passed_verification_device, GSPAR_PARAM_PRESENT);
		kernel_rank_7->setParameter("iteration", iteration);
		kernel_rank_7->setParameter("TEST_ARRAY_SIZE", TEST_ARRAY_SIZE);
		kernel_rank_7->setParameter("CLASS", CLASS);
		kernel_rank_7->setParameter("NUM_KEYS", NUM_KEYS);
		kernel_rank_7->setNumThreadsPerBlockForX(threads_per_block_on_rank_7);
		kernel_rank_7->addExtraKernelCode(source_additional_routines);
		kernel_rank_7->compile<Instance>(dims);
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}
}

std::string source_kernel_rank_1 = GSPAR_STRINGIZE_SOURCE(
	key_array[iteration] = iteration;
	key_array[iteration+MAX_ITERATIONS] = MAX_KEY - iteration;
	for(int i=0; i<TEST_ARRAY_SIZE; i++){
		partial_verify_vals[i] = key_array[test_index_array[i]];
	}
);

std::string source_kernel_rank_2 = GSPAR_STRINGIZE_SOURCE(
	int thread_id = gspar_get_global_id(0);

	key_buff1[thread_id] = 0;
);

std::string source_kernel_rank_3 = GSPAR_STRINGIZE_SOURCE(
	int thread_id = gspar_get_global_id(0);

	gspar_atomic_add_int(&key_buff1[key_array[thread_id]], 1);
);

std::string source_kernel_rank_4 = GSPAR_STRINGIZE_SOURCE(
	int thread_id = gspar_get_global_id(0);

	GSPAR_DEVICE_SHARED_MEMORY int shared_data[2*IS_THREADS_PER_BLOCK_ON_RANK];

	int* destiny = source;

	shared_data[gspar_get_thread_id(0)] = 0;
	int position = gspar_get_block_size(0) + gspar_get_thread_id(0);

	int factor = MAX_KEY / number_of_blocks;
	int start = factor * gspar_get_block_id(0);
	int end = start + factor;

	for(int i=start; i<end; i+=gspar_get_block_size(0)){
		shared_data[position] = source[i + gspar_get_thread_id(0)];

		for(int offset=1; offset<gspar_get_block_size(0); offset<<=1){
			gspar_synchronize_local_threads();
			int t = shared_data[position] + shared_data[position - offset];
			gspar_synchronize_local_threads();
			shared_data[position] = t;
		}

		int prv_val = (i == start) ? 0 : destiny[i - 1];
		destiny[i + gspar_get_thread_id(0)] = shared_data[position] + prv_val;
	}

	gspar_synchronize_local_threads();
	if(gspar_get_thread_id(0)==0){sum[gspar_get_block_id(0)]=destiny[end-1];}
);

std::string source_kernel_rank_5 = GSPAR_STRINGIZE_SOURCE(
	GSPAR_DEVICE_SHARED_MEMORY int shared_data[2*IS_THREADS_PER_BLOCK_ON_RANK];

	shared_data[gspar_get_thread_id(0)] = 0;
	int position = gspar_get_block_size(0) + gspar_get_thread_id(0);
	shared_data[position] = source[gspar_get_thread_id(0)];

	for(int offset=1; offset<gspar_get_block_size(0); offset<<=1){
		gspar_synchronize_local_threads();
		int t = shared_data[position] + shared_data[position - offset];
		gspar_synchronize_local_threads();
		shared_data[position] = t;
	}

	gspar_synchronize_local_threads();

	destiny[gspar_get_thread_id(0)] = shared_data[position - 1];
);

std::string source_kernel_rank_6 = GSPAR_STRINGIZE_SOURCE(
	int* destiny = source;
	int factor = MAX_KEY / number_of_blocks;
	int start = factor * gspar_get_block_id(0);
	int end = start + factor;
	int sum = offset[gspar_get_block_id(0)];
	for(int i=start; i<end; i+=gspar_get_block_size(0)){
		destiny[i + gspar_get_thread_id(0)] = source[i + gspar_get_thread_id(0)] + sum;
	}
);

std::string source_kernel_rank_7 = GSPAR_STRINGIZE_SOURCE(
	int i, k;
	int passed_verification = 0;
	for(i=0; i<TEST_ARRAY_SIZE; i++){                                         
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
);

std::string source_additional_routines =
    "#define IS_THREADS_PER_BLOCK_ON_RANK " + std::to_string(IS_THREADS_PER_BLOCK_ON_RANK) + "\n";
