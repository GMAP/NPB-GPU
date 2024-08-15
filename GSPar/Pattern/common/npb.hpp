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
 */
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>
#include <cmath>
#include <string>

#if defined(GSPARDRIVER_CUDA)
#include "GSPar_CUDA.hpp"
using namespace GSPar::Driver::CUDA;
#elif defined(GSPARDRIVER_OPENCL)
#include "GSPar_OpenCL.hpp"
using namespace GSPar::Driver::OpenCL;
#endif

#if defined(USE_POW)
#define r23 pow(0.5, 23.0)
#define r46 (r23*r23)
#define t23 pow(2.0, 23.0)
#define t46 (t23*t23)
#else
#define r23 (0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5)
#define r46 (r23*r23)
#define t23 (2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0)
#define t46 (t23*t23)
#endif

#define PROFILING_SLOTS 64

typedef int boolean;
//typedef struct { double real; double imag; } dcomplex;
typedef struct tdcomplex { double real; double imag; } dcomplex;

#define TRUE  1
#define FALSE 0

#define npb_max(a,b) (((a) > (b)) ? (a) : (b))
#define npb_min(a,b) (((a) < (b)) ? (a) : (b))
#define	pow2(a) ((a)*(a))

/* old version of the complex number operations */
#define get_real(c) c.real
#define get_imag(c) c.imag
#define cadd(c,a,b) (c.real = a.real + b.real, c.imag = a.imag + b.imag)
#define csub(c,a,b) (c.real = a.real - b.real, c.imag = a.imag - b.imag)
#define cmul(c,a,b) (c.real = a.real * b.real - a.imag * b.imag, \
		c.imag = a.real * b.imag + a.imag * b.real)
#define crmul(c,a,b) (c.real = a.real * b, c.imag = a.imag * b)

/* latest version of the complex number operations */
#define dcomplex_create(r,i) (dcomplex){r, i}
#define dcomplex_add(a,b) (dcomplex){(a).real+(b).real, (a).imag+(b).imag}
#define dcomplex_sub(a,b) (dcomplex){(a).real-(b).real, (a).imag-(b).imag}
#define dcomplex_mul(a,b) (dcomplex){((a).real*(b).real)-((a).imag*(b).imag),\
	((a).real*(b).imag)+((a).imag*(b).real)}
#define dcomplex_mul2(a,b) (dcomplex){(a).real*(b), (a).imag*(b)}
static inline dcomplex dcomplex_div(dcomplex z1, dcomplex z2){
	double a = z1.real;
	double b = z1.imag;
	double c = z2.real;
	double d = z2.imag;
	double divisor = c*c + d*d;
	double real = (a*c + b*d) / divisor;
	double imag = (b*c - a*d) / divisor;
	dcomplex result = (dcomplex){real, imag};
	return result;
}
#define dcomplex_div2(a,b) (dcomplex){(a).real/(b), (a).imag/(b)}
#define dcomplex_abs(x)    sqrt(((x).real*(x).real) + ((x).imag*(x).imag))
#define dconjg(x)          (dcomplex){(x).real, -1.0*(x).imag}

double start[PROFILING_SLOTS], elapsed[PROFILING_SLOTS];

extern double randlc(double*, double);
extern void vranlc(int, double*, double, double*);

extern void wtime(double*);
extern double elapsed_time(void);
extern void timer_clear(int);
extern void timer_start(int);
extern void timer_stop(int);
extern double timer_read(int);

extern void c_print_results(char* name,
		char class_npb,
		int n1,
		int n2,
		int n3,
		int niter,
		double t,
		double mops,
		char* optype,
		int passed_verification,
		char* npbversion,
		char* compiletime,
		char* cc,
		char* clink,
		char* c_lib,
		char* c_inc,
		char* cflags,
		char* clinkflags,
		char* rand);

extern void c_print_results(char* name,
		char class_npb,
		int n1,
		int n2,
		int n3,
		int niter,
		double t,
		double mops,
		char* optype,
		int passed_verification,
		char* npbversion,
		char* compiletime,
		char* cc,
		char* clink,
		char* c_lib,
		char* c_inc,
		char* cflags,
		char* clinkflags,
		char* rand,
		long long int offloading);

extern void c_print_results(char* name,
		char class_npb,
		int n1,
		int n2,
		int n3,
		int niter,
		double t,
		double mops,
		char* optype,
		int passed_verification,
		char* npbversion,
		char* compiletime,
		char* compilerversion,
		char* libversion,
		char* cpu_device,
		char* gpu_device,
		char* gpu_config,
		char* cc,
		char* clink,
		char* c_lib,
		char* c_inc,
		char* cflags,
		char* clinkflags,
		char* rand,
		long long int offloading);

extern void c_print_results(char* name,
		char class_npb,
		int n1, 
		int n2,
		int n3,
		int niter,
		double t,
		double mops,
		char* optype,
		int passed_verification,
		char* npbversion,
		char* compiletime,
		char* compilerversion,
		char* libversion,
		char* cpu_device,
		char* gpu_device,
		//char* gpu_config,
		char* cc,
		char* clink,
		char* c_lib,
		char* c_inc,
		char* cflags,
		char* clinkflags,
		char* rand);		

/*
 * ---------------------------------------------------------------------
 *
 * this routine returns a uniform pseudorandom double precision number in the
 * range (0, 1) by using the linear congruential generator
 * 
 * x_{k+1} = a x_k  (mod 2^46)
 *
 * where 0 < x_k < 2^46 and 0 < a < 2^46. this scheme generates 2^44 numbers
 * before repeating. the argument A is the same as 'a' in the above formula,
 * and X is the same as x_0.  A and X must be odd double precision integers
 * in the range (1, 2^46). the returned value RANDLC is normalized to be
 * between 0 and 1, i.e. RANDLC = 2^(-46) * x_1.  X is updated to contain
 * the new seed x_1, so that subsequent calls to RANDLC using the same
 * arguments will generate a continuous sequence.
 * 
 * this routine should produce the same results on any computer with at least
 * 48 mantissa bits in double precision floating point data.  On 64 bit
 * systems, double precision should be disabled.
 *
 * David H. Bailey, October 26, 1990
 * 
 * ---------------------------------------------------------------------
 */
double randlc(double* x, double a){    
	double t1,t2,t3,t4,a1,a2,x1,x2,z;

	/*
	 * ---------------------------------------------------------------------
	 * break A into two parts such that A = 2^23 * A1 + A2.
	 * ---------------------------------------------------------------------
	 */
	t1 = r23 * a;
	a1 = (int)t1;
	a2 = a - t23 * a1;

	/*
	 * ---------------------------------------------------------------------
	 * break X into two parts such that X = 2^23 * X1 + X2, compute
	 * Z = A1 * X2 + A2 * X1  (mod 2^23), and then
	 * X = 2^23 * Z + A2 * X2  (mod 2^46).
	 * ---------------------------------------------------------------------
	 */
	t1 = r23 * (*x);
	x1 = (int)t1;
	x2 = (*x) - t23 * x1;
	t1 = a1 * x2 + a2 * x1;
	t2 = (int)(r23 * t1);
	z = t1 - t23 * t2;
	t3 = t23 * z + a2 * x2;
	t4 = (int)(r46 * t3);
	(*x) = t3 - t46 * t4;

	return (r46 * (*x));
}

/*
 * ---------------------------------------------------------------------
 *
 * this routine generates N uniform pseudorandom double precision numbers in
 * the range (0, 1) by using the linear congruential generator
 *
 * x_{k+1} = a x_k  (mod 2^46)
 *
 * where 0 < x_k < 2^46 and 0 < a < 2^46. this scheme generates 2^44 numbers
 * before repeating. the argument A is the same as 'a' in the above formula,
 * and X is the same as x_0. A and X must be odd double precision integers
 * in the range (1, 2^46). the N results are placed in Y and are normalized
 * to be between 0 and 1. X is updated to contain the new seed, so that
 * subsequent calls to VRANLC using the same arguments will generate a
 * continuous sequence.  if N is zero, only initialization is performed, and
 * the variables X, A and Y are ignored.
 *
 * this routine is the standard version designed for scalar or RISC systems.
 * however, it should produce the same results on any single processor
 * computer with at least 48 mantissa bits in double precision floating point
 * data. on 64 bit systems, double precision should be disabled.
 *
 * ---------------------------------------------------------------------
 */
void vranlc(int n, double* x_seed, double a, double y[]){
	int i;
	double x,t1,t2,t3,t4,a1,a2,x1,x2,z;

	/*
	 * ---------------------------------------------------------------------
	 * break A into two parts such that A = 2^23 * A1 + A2.
	 * ---------------------------------------------------------------------
	 */
	t1 = r23 * a;
	a1 = (int)t1;
	a2 = a - t23 * a1;
	x = *x_seed;

	/*
	 * ---------------------------------------------------------------------
	 * generate N results. this loop is not vectorizable.
	 * ---------------------------------------------------------------------
	 */
	for(i=0; i<n; i++){
		/*
		 * ---------------------------------------------------------------------
		 * break X into two parts such that X = 2^23 * X1 + X2, compute
		 * Z = A1 * X2 + A2 * X1  (mod 2^23), and then
		 * X = 2^23 * Z + A2 * X2  (mod 2^46).
		 * ---------------------------------------------------------------------
		 */
		t1 = r23 * x;
		x1 = (int)t1;
		x2 = x - t23 * x1;
		t1 = a1 * x2 + a2 * x1;
		t2 = (int)(r23 * t1);
		z = t1 - t23 * t2;
		t3 = t23 * z + a2 * x2;
		t4 = (int)(r46 * t3);
		x = t3 - t46 * t4;
		y[i] = r46 * x;
	}
	*x_seed = x;
}

void wtime(double *t){
	static int sec = -1;
	struct timeval tv;
	gettimeofday(&tv, 0);
	if (sec < 0) sec = tv.tv_sec;
	*t = (tv.tv_sec - sec) + 1.0e-6*tv.tv_usec;
}

/*****************************************************************/
/******         E  L  A  P  S  E  D  _  T  I  M  E          ******/
/*****************************************************************/
double elapsed_time(void){
	double t;
	wtime(&t);
	return(t);
}

/*****************************************************************/
/******            T  I  M  E  R  _  C  L  E  A  R          ******/
/*****************************************************************/
void timer_clear(int n){
	elapsed[n] = 0.0;
}

/*****************************************************************/
/******            T  I  M  E  R  _  S  T  A  R  T          ******/
/*****************************************************************/
void timer_start(int n){
	start[n] = elapsed_time();
}

/*****************************************************************/
/******            T  I  M  E  R  _  S  T  O  P             ******/
/*****************************************************************/
void timer_stop(int n){
	double t, now;
	now = elapsed_time();
	t = now - start[n];
	elapsed[n] += t;
}

/*****************************************************************/
/******            T  I  M  E  R  _  R  E  A  D             ******/
/*****************************************************************/
double timer_read(int n){
	return(elapsed[n]);
}

/*****************************************************************/
/******     C  _  P  R  I  N  T  _  R  E  S  U  L  T  S     ******/
/*****************************************************************/
void c_print_results(char* name,
		char class_npb,
		int n1, 
		int n2,
		int n3,
		int niter,
		double t,
		double mops,
		char* optype,
		int passed_verification,
		char* npbversion,
		char* compiletime,
		char* cc,
		char* clink,
		char* c_lib,
		char* c_inc,
		char* cflags,
		char* clinkflags,
		char* rand){
	printf("\n\n %s Benchmark Completed\n", name);
	printf(" class_npb       =                        %c\n", class_npb);
	if((name[0]=='I')&&(name[1]=='S')){
		if(n3==0){
			long nn = n1;
			if(n2!=0){nn*=n2;}
			printf(" Size            =             %12ld\n", nn); /* as in IS */
		}else{
			printf(" Size            =             %4dx%4dx%4d\n", n1,n2,n3);
		}
	}else{
		char size[16];
		int j;
		if((n2==0) && (n3==0)){
			if((name[0]=='E')&&(name[1]=='P')){
				sprintf(size, "%15.0lf", pow(2.0, n1));
				j = 14;
				if(size[j] == '.'){
					size[j] = ' '; 
					j--;
				}
				size[j+1] = '\0';
				printf(" Size            =          %15s\n", size);
			}else{
				printf(" Size            =             %12d\n", n1);
			}
		}else{
			printf(" Size            =           %4dx%4dx%4d\n", n1, n2, n3);
		}
	}	
	printf(" Iterations      =             %12d\n", niter); 
	printf(" Time in seconds =             %12.2f\n", t);
	printf(" Mop/s total     =             %12.2f\n", mops);
	printf(" Operation type  = %24s\n", optype);
	if(passed_verification < 0){
		printf( " Verification    =            NOT PERFORMED\n");
	}else if(passed_verification){
		printf(" Verification    =               SUCCESSFUL\n");
	}else{
		printf(" Verification    =             UNSUCCESSFUL\n");
	}
	printf(" Version         =             %12s\n", npbversion);
	printf(" Compile date    =             %12s\n", compiletime);
	printf("\n Compile options:\n");
	printf("    CC           = %s\n", cc);
	printf("    CLINK        = %s\n", clink);
	printf("    C_LIB        = %s\n", c_lib);
	printf("    C_INC        = %s\n", c_inc);
	printf("    CFLAGS       = %s\n", cflags);
	printf("    CLINKFLAGS   = %s\n", clinkflags);
	printf("    RAND         = %s\n", rand);
#ifdef SMP
	evalue = getenv("MP_SET_NUMTHREADS");
	printf("   MULTICPUS = %s\n", evalue);
#endif    
	/* 
	 * printf(" Please send the results of this run to:\n\n");
	 * printf(" NPB Development Team\n");
	 * printf(" Internet: npb@nas.nasa.gov\n \n");
	 * printf(" If email is not available, send this to:\n\n");
	 * printf(" MS T27A-1\n");
	 * printf(" NASA Ames Research Center\n");
	 * printf(" Moffett Field, CA  94035-1000\n\n");
	 * printf(" Fax: 650-604-3957\n\n");
	 */
	printf("\n\n");
	printf("----------------------------------------------------------------------\n");
	printf(" NPB-CPP is developed by:\n");
	printf("            Dalvan Griebler <dalvangriebler@gmail.com>\n");
	printf("            Gabriell Araujo <hexenoften@gmail.com>\n");
	printf("            Júnior Löff <loffjh@gmail.com>\n");
	printf("\n");
	printf(" In case of problems, send an email to us\n");
	printf("----------------------------------------------------------------------\n");
	printf("\n");
}

/*****************************************************************/
/******     C  _  P  R  I  N  T  _  R  E  S  U  L  T  S     ******/
/*****************************************************************/
void c_print_results(char* name,
		char class_npb,
		int n1, 
		int n2,
		int n3,
		int niter,
		double t,
		double mops,
		char* optype,
		int passed_verification,
		char* npbversion,
		char* compiletime,
		char* cc,
		char* clink,
		char* c_lib,
		char* c_inc,
		char* cflags,
		char* clinkflags,
		char* rand,
		long long int offloading){
	printf("\n\n %s Benchmark Completed\n", name);
	printf(" class_npb       =                        %c\n", class_npb);
	if((name[0]=='I')&&(name[1]=='S')){
		if(n3==0){
			long nn = n1;
			if(n2!=0){nn*=n2;}
			printf(" Size            =             %12ld\n", nn); /* as in IS */
		}else{
			printf(" Size            =             %4dx%4dx%4d\n", n1,n2,n3);
		}
	}else{
		char size[16];
		int j;
		if((n2==0) && (n3==0)){
			if((name[0]=='E')&&(name[1]=='P')){
				sprintf(size, "%15.0lf", pow(2.0, n1));
				j = 14;
				if(size[j] == '.'){
					size[j] = ' '; 
					j--;
				}
				size[j+1] = '\0';
				printf(" Size            =          %15s\n", size);
			}else{
				printf(" Size            =             %12d\n", n1);
			}
		}else{
			printf(" Size            =           %4dx%4dx%4d\n", n1, n2, n3);
		}
	}	
	printf(" Iterations      =             %12d\n", niter); 
	printf(" Offloading      =             %12lld\n", offloading);
	printf(" Time in seconds =             %12.2f\n", t);
	printf(" Mop/s total     =             %12.2f\n", mops);
	printf(" Operation type  = %24s\n", optype);
	if(passed_verification < 0){
		printf( " Verification    =            NOT PERFORMED\n");
	}else if(passed_verification){
		printf(" Verification    =               SUCCESSFUL\n");
	}else{
		printf(" Verification    =             UNSUCCESSFUL\n");
	}
	printf(" Version         =             %12s\n", npbversion);
	printf(" Compile date    =             %12s\n", compiletime);
	printf("\n Compile options:\n");
	printf("    CC           = %s\n", cc);
	printf("    CLINK        = %s\n", clink);
	printf("    C_LIB        = %s\n", c_lib);
	printf("    C_INC        = %s\n", c_inc);
	printf("    CFLAGS       = %s\n", cflags);
	printf("    CLINKFLAGS   = %s\n", clinkflags);
	printf("    RAND         = %s\n", rand);
#ifdef SMP
	evalue = getenv("MP_SET_NUMTHREADS");
	printf("   MULTICPUS = %s\n", evalue);
#endif    
	/* 
	 * printf(" Please send the results of this run to:\n\n");
	 * printf(" NPB Development Team\n");
	 * printf(" Internet: npb@nas.nasa.gov\n \n");
	 * printf(" If email is not available, send this to:\n\n");
	 * printf(" MS T27A-1\n");
	 * printf(" NASA Ames Research Center\n");
	 * printf(" Moffett Field, CA  94035-1000\n\n");
	 * printf(" Fax: 650-604-3957\n\n");
	 */
	printf("\n\n");
	printf("----------------------------------------------------------------------\n");
	printf(" NPB-CPP is developed by:\n");
	printf("            Dalvan Griebler <dalvangriebler@gmail.com>\n");
	printf("            Gabriell Araujo <hexenoften@gmail.com>\n");
	printf("            Júnior Löff <loffjh@gmail.com>\n");
	printf("\n");
	printf(" In case of problems, send an email to us\n");
	printf("----------------------------------------------------------------------\n");
	printf("\n");
}

/*****************************************************************/
/******     C  _  P  R  I  N  T  _  R  E  S  U  L  T  S     ******/
/*****************************************************************/
void c_print_results(char* name,
		char class_npb,
		int n1, 
		int n2,
		int n3,
		int niter,
		double t,
		double mops,
		char* optype,
		int passed_verification,
		char* npbversion,
		char* compiletime,
		char* compilerversion,
		char* libversion,
		char* cpu_device,
		char* gpu_device,
		char* gpu_config,
		char* cc,
		char* clink,
		char* c_lib,
		char* c_inc,
		char* cflags,
		char* clinkflags,
		char* rand,
		long long int offloading){
			printf("\n\n %s Benchmark Completed\n", name);
			//printf(" class_npb       =                        %c\n", class_npb);
			printf(" Class           =                        %c\n", class_npb);
			if((name[0]=='I')&&(name[1]=='S')){
				if(n3==0){
					long nn = n1;
					if(n2!=0){nn*=n2;}
					printf(" Size            =             %12ld\n", nn); /* as in IS */
				}else{
					printf(" Size            =             %4dx%4dx%4d\n", n1,n2,n3);
				}
			}else{
				char size[16];
				int j;
				if((n2==0) && (n3==0)){
					if((name[0]=='E')&&(name[1]=='P')){
						sprintf(size, "%15.0lf", pow(2.0, n1));
						j = 14;
						if(size[j] == '.'){
							size[j] = ' '; 
							j--;
						}
						size[j+1] = '\0';
						printf(" Size            =          %15s\n", size);
					}else{
						printf(" Size            =             %12d\n", n1);
					}
				}else{
					printf(" Size            =           %4dx%4dx%4d\n", n1, n2, n3);
				}
			}	
			printf(" Iterations      =             %12d\n", niter); 
			printf(" Offloading      =             %12lld\n", offloading);
			printf(" Time in seconds =             %12.2f\n", t);
			printf(" Mop/s total     =             %12.2f\n", mops);
			printf(" Operation type  = %24s\n", optype);
			if(passed_verification < 0){
				printf( " Verification    =            NOT PERFORMED\n");
			}else if(passed_verification){
				printf(" Verification    =               SUCCESSFUL\n");
			}else{
				printf(" Verification    =             UNSUCCESSFUL\n");
			}
			printf(" Version         =             %12s\n", npbversion);
			printf(" Compile date    =             %12s\n", compiletime);
			//printf(" NVCC version    =             %12s\n", compilerversion);
			printf(" CUDA version    =             %12s\n", libversion);
			printf("\n Compile options:\n");
			printf("    CC           = %s\n", cc);
			printf("    CLINK        = %s\n", clink);
			printf("    C_LIB        = %s\n", c_lib);
			printf("    C_INC        = %s\n", c_inc);
			printf("    CFLAGS       = %s\n", cflags);
			printf("    CLINKFLAGS   = %s\n", clinkflags);
			printf("    RAND         = %s\n", rand);
			printf("\n Hardware:\n");
			printf("    CPU device   = %s\n", cpu_device);
			printf("    GPU device   = %s\n", gpu_device);
			printf("\n Software:\n");
			printf("    Profiling    = %s\n", gpu_config);
#ifdef SMP
			evalue = getenv("MP_SET_NUMTHREADS");
			printf("   MULTICPUS = %s\n", evalue);
#endif    
			/* 
			 * printf(" Please send the results of this run to:\n\n");
			 * printf(" NPB Development Team\n");
			 * printf(" Internet: npb@nas.nasa.gov\n \n");
			 * printf(" If email is not available, send this to:\n\n");
			 * printf(" MS T27A-1\n");
			 * printf(" NASA Ames Research Center\n");
			 * printf(" Moffett Field, CA  94035-1000\n\n");
			 * printf(" Fax: 650-604-3957\n\n");
			 */
			printf("\n");
			printf("----------------------------------------------------------------------\n");
			printf(" NPB-CPP is developed by:\n");
			printf("            Dalvan Griebler <dalvangriebler@gmail.com>\n");
			printf("            Gabriell Araujo <hexenoften@gmail.com>\n");
			printf("            Júnior Löff <loffjh@gmail.com>\n");
			printf("\n");
			printf(" In case of problems, send an email to us\n");
			printf("----------------------------------------------------------------------\n");
			printf("\n");
}

/*****************************************************************/
/******     C  _  P  R  I  N  T  _  R  E  S  U  L  T  S     ******/
/*****************************************************************/
void c_print_results(char* name,
		char class_npb,
		int n1, 
		int n2,
		int n3,
		int niter,
		double t,
		double mops,
		char* optype,
		int passed_verification,
		char* npbversion,
		char* compiletime,
		char* compilerversion,
		char* libversion,
		char* cpu_device,
		char* gpu_device,
		char* gpu_config,
		char* cc,
		char* clink,
		char* c_lib,
		char* c_inc,
		char* cflags,
		char* clinkflags,
		char* rand){
			printf("\n\n %s Benchmark Completed\n", name);
			//printf(" class_npb       =                        %c\n", class_npb);
			printf(" Class           =                        %c\n", class_npb);
			if((name[0]=='I')&&(name[1]=='S')){
				if(n3==0){
					long nn = n1;
					if(n2!=0){nn*=n2;}
					printf(" Size            =             %12ld\n", nn); /* as in IS */
				}else{
					printf(" Size            =             %4dx%4dx%4d\n", n1,n2,n3);
				}
			}else{
				char size[16];
				int j;
				if((n2==0) && (n3==0)){
					if((name[0]=='E')&&(name[1]=='P')){
						sprintf(size, "%15.0lf", pow(2.0, n1));
						j = 14;
						if(size[j] == '.'){
							size[j] = ' '; 
							j--;
						}
						size[j+1] = '\0';
						printf(" Size            =          %15s\n", size);
					}else{
						printf(" Size            =             %12d\n", n1);
					}
				}else{
					printf(" Size            =           %4dx%4dx%4d\n", n1, n2, n3);
				}
			}	
			printf(" Iterations      =             %12d\n", niter); 
			printf(" Time in seconds =             %12.2f\n", t);
			printf(" Mop/s total     =             %12.2f\n", mops);
			printf(" Operation type  = %24s\n", optype);
			if(passed_verification < 0){
				printf( " Verification    =            NOT PERFORMED\n");
			}else if(passed_verification){
				printf(" Verification    =               SUCCESSFUL\n");
			}else{
				printf(" Verification    =             UNSUCCESSFUL\n");
			}
			printf(" Version         =             %12s\n", npbversion);
			printf(" Compile date    =             %12s\n", compiletime);
			//printf(" NVCC version    =             %12s\n", compilerversion);
			printf(" CUDA version    =             %12s\n", libversion);
			printf("\n Compile options:\n");
			printf("    CC           = %s\n", cc);
			printf("    CLINK        = %s\n", clink);
			printf("    C_LIB        = %s\n", c_lib);
			printf("    C_INC        = %s\n", c_inc);
			printf("    CFLAGS       = %s\n", cflags);
			printf("    CLINKFLAGS   = %s\n", clinkflags);
			printf("    RAND         = %s\n", rand);
			printf("\n Hardware:\n");
			printf("    CPU device   = %s\n", cpu_device);
			printf("    GPU device   = %s\n", gpu_device);
			printf("\n Software:\n");
			printf("    Profiling    = %s\n", gpu_config);
#ifdef SMP
			evalue = getenv("MP_SET_NUMTHREADS");
			printf("   MULTICPUS = %s\n", evalue);
#endif    
			/* 
			 * printf(" Please send the results of this run to:\n\n");
			 * printf(" NPB Development Team\n");
			 * printf(" Internet: npb@nas.nasa.gov\n \n");
			 * printf(" If email is not available, send this to:\n\n");
			 * printf(" MS T27A-1\n");
			 * printf(" NASA Ames Research Center\n");
			 * printf(" Moffett Field, CA  94035-1000\n\n");
			 * printf(" Fax: 650-604-3957\n\n");
			 */
			printf("\n");
			printf("----------------------------------------------------------------------\n");
			printf(" NPB-CPP is developed by:\n");
			printf("            Dalvan Griebler <dalvangriebler@gmail.com>\n");
			printf("            Gabriell Araujo <hexenoften@gmail.com>\n");
			printf("            Júnior Löff <loffjh@gmail.com>\n");
			printf("\n");
			printf(" In case of problems, send an email to us\n");
			printf("----------------------------------------------------------------------\n");
			printf("\n");
}

/*****************************************************************/
/******     C  _  P  R  I  N  T  _  R  E  S  U  L  T  S     ******/
/*****************************************************************/
void c_print_results(char* name,
		char class_npb,
		int n1, 
		int n2,
		int n3,
		int niter,
		double t,
		double mops,
		char* optype,
		int passed_verification,
		char* npbversion,
		char* compiletime,
		char* compilerversion,
		char* libversion,
		char* cpu_device,
		char* gpu_device,
		char* cc,
		char* clink,
		char* c_lib,
		char* c_inc,
		char* cflags,
		char* clinkflags,
		char* rand){
			printf("\n\n %s Benchmark Completed\n", name);
			//printf(" class_npb       =                        %c\n", class_npb);
			printf(" Class           =                        %c\n", class_npb);
			if((name[0]=='I')&&(name[1]=='S')){
				if(n3==0){
					long nn = n1;
					if(n2!=0){nn*=n2;}
					printf(" Size            =             %12ld\n", nn); /* as in IS */
				}else{
					printf(" Size            =             %4dx%4dx%4d\n", n1,n2,n3);
				}
			}else{
				char size[16];
				int j;
				if((n2==0) && (n3==0)){
					if((name[0]=='E')&&(name[1]=='P')){
						sprintf(size, "%15.0lf", pow(2.0, n1));
						j = 14;
						if(size[j] == '.'){
							size[j] = ' '; 
							j--;
						}
						size[j+1] = '\0';
						printf(" Size            =          %15s\n", size);
					}else{
						printf(" Size            =             %12d\n", n1);
					}
				}else{
					printf(" Size            =           %4dx%4dx%4d\n", n1, n2, n3);
				}
			}	
			printf(" Iterations      =             %12d\n", niter); 
			printf(" Time in seconds =             %12.2f\n", t);
			printf(" Mop/s total     =             %12.2f\n", mops);
			printf(" Operation type  = %24s\n", optype);
			if(passed_verification < 0){
				printf( " Verification    =            NOT PERFORMED\n");
			}else if(passed_verification){
				printf(" Verification    =               SUCCESSFUL\n");
			}else{
				printf(" Verification    =             UNSUCCESSFUL\n");
			}
			printf(" Version         =             %12s\n", npbversion);
			printf(" Compile date    =             %12s\n", compiletime);
			//printf(" NVCC version    =             %12s\n", compilerversion);
			printf(" CUDA version    =             %12s\n", libversion);
			printf("\n Compile options:\n");
			printf("    CC           = %s\n", cc);
			printf("    CLINK        = %s\n", clink);
			printf("    C_LIB        = %s\n", c_lib);
			printf("    C_INC        = %s\n", c_inc);
			printf("    CFLAGS       = %s\n", cflags);
			printf("    CLINKFLAGS   = %s\n", clinkflags);
			printf("    RAND         = %s\n", rand);
			printf("\n Hardware:\n");
			printf("    CPU device   = %s\n", cpu_device);
			printf("    GPU device   = %s\n", gpu_device);
			//printf("\n Software:\n");
			//printf("    Profiling    = %s\n", gpu_config);
#ifdef SMP
			evalue = getenv("MP_SET_NUMTHREADS");
			printf("   MULTICPUS = %s\n", evalue);
#endif    
			/* 
			 * printf(" Please send the results of this run to:\n\n");
			 * printf(" NPB Development Team\n");
			 * printf(" Internet: npb@nas.nasa.gov\n \n");
			 * printf(" If email is not available, send this to:\n\n");
			 * printf(" MS T27A-1\n");
			 * printf(" NASA Ames Research Center\n");
			 * printf(" Moffett Field, CA  94035-1000\n\n");
			 * printf(" Fax: 650-604-3957\n\n");
			 */
			printf("\n");
			printf("----------------------------------------------------------------------\n");
			printf(" NPB-CPP is developed by:\n");
			printf("            Dalvan Griebler <dalvangriebler@gmail.com>\n");
			printf("            Gabriell Araujo <hexenoften@gmail.com>\n");
			printf("            Júnior Löff <loffjh@gmail.com>\n");
			printf("\n");
			printf(" In case of problems, send an email to us\n");
			printf("----------------------------------------------------------------------\n");
			printf("\n");
}