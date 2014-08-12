/*
 * phasediagram.cpp
 *
 *  Created on: Aug 12, 2014
 *      Author: Abuenameh
 */

#include <ctime>

#include "L-BFGS/lbfgsb.h"

#define L 3
#define nmax 3

real* f_tb_host;
real* f_tb_dev;

cublasHandle_t cublasHd;
real stpscal;

real U;
real J;
real mu;

void funcgrad(real* x, real& f, real* g, const cudaStream_t& stream) {
//	energy(x, f_tb_dev, g, L, nmax, U, J, mu, stream);
	f = *f_tb_host;
}



int main() {

	time_t start = time(NULL);

	int dim = L * (nmax + 1);

	const real epsg = EPSG;
	const real epsf = EPSF;
	const real epsx = EPSX;
	const int maxits = MAXITS;
	stpscal = 1;
	int info;

	real* x;
	int* nbd;
	real* l;
	real* u;
	memAlloc<real>(&x, dim);
	memAlloc<int>(&nbd, dim);
	memAlloc<real>(&l, dim);
	memAlloc<real>(&u, dim);
	memAllocHost<real>(&f_tb_host, &f_tb_dev, 1);

	cudaSetDeviceFlags(cudaDeviceMapHost);
	cublasCreate_v2(&cublasHd);

	U = 1;
	J = 0.2;
	mu = 0.5;

//	for(int i = 0; i < 10; i++) {
//	initProb(x, nbd, l, u, L, nmax);
//	lbfgsbminimize(dim, 4, x, epsg, epsf, epsx, maxits, nbd, l, u, info);
	printf("info: %d\n", info);
//	printf("f: %.20e\n", *f_tb_host);
//	}

	printf("f: %.20e\n", *f_tb_host);
	real* x_host = new real[dim];
	memCopy(x_host, x, dim * sizeof(real), cudaMemcpyDeviceToHost);
	printf("x: ");
	for (int i = 0; i < dim; i++) {
		printf("%f, ", x_host[i]);
	}
	printf("\n");

	memFreeHost(f_tb_host);
	memFree(x);
	memFree(nbd);
	memFree(l);
	memFree(u);

	cublasDestroy_v2(cublasHd);

	cudaDeviceReset();

	time_t end = time(NULL);

	printf("Runtime: %ld", end-start);
}

