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
double randlc(double *x, double a){    
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
void vranlc(int n, double *x_seed, double a, double y[]){
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
