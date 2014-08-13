/*
 * phasediagram.cpp
 *
 *  Created on: Aug 12, 2014
 *      Author: Abuenameh
 */

#include <ctime>
#include <vector>

#include "L-BFGS/lbfgsb.h"

#include "phasediagram.hpp"

using std::vector;

//#define L 3
//#define nmax 3

real* f_tb_host;
real* f_tb_dev;

cublasHandle_t cublasHd;
real stpscal;

real U;
real J;
real mu;

extern void initProb(real* x, int* nbd, real* l, real* u);
extern void initProb(real* x, int* nbd, real* l, real* u, int i, real dx);

void funcgrad(real* x, real& f, real* g, const cudaStream_t& stream) {
//	energy(x, f_tb_dev, g, L, nmax, U, J, mu, stream);
	f = *f_tb_host;
}

void energy(real* x, real* f_dev, real* g_dev, real* U, real* J, real mu,
	real* norm2s);

int main() {

	time_t start = time(NULL);

	int ndim = 2 * L * dim;

	const real epsg = EPSG;
	const real epsf = EPSF;
	const real epsx = EPSX;
	const int maxits = MAXITS;
	stpscal = 1;
	int info;

	cudaSetDeviceFlags(cudaDeviceMapHost);
	cublasCreate_v2(&cublasHd);

	real* x;
	int* nbd;
	real* l;
	real* u;
	memAlloc<real>(&x, ndim);
	memAlloc<int>(&nbd, ndim);
	memAlloc<real>(&l, ndim);
	memAlloc<real>(&u, ndim);
	memAllocHost<real>(&f_tb_host, &f_tb_dev, 1);

	vector<real> x_host(ndim);
	vector<real> norm2_host(L, 0);

	U = 1;
	J = 0.2;
	mu = 0.5;

//	for(int i = 0; i < 10; i++) {

	printf("Before initProb\n");
	initProb(x, nbd, l, u);
	memCopy(x_host.data(), x, ndim*sizeof(real), cudaMemcpyDeviceToHost);
	for(int i = 0; i < L; i++) {
		norm2_host[i] = 0;
		for(int n = 0; n <= nmax; n++) {
			norm2_host[i] += x_host[2*(i*dim+n)]*x_host[2*(i*dim+n)]+x_host[2*(i*dim+n)+1]*x_host[2*(i*dim+n)+1];
		}
//		printf("norm2[%d] = %f\n", i, norm2_host[i]);
	}

//	lbfgsbminimize(dim, 4, x, epsg, epsf, epsx, maxits, nbd, l, u, info);
	printf("info: %d\n", info);
//	printf("f: %.20e\n", *f_tb_host);
//	}

	real* U;
	real* J;
	memAlloc<real>(&J, L);
	memAlloc<real>(&U, L);
	vector<real> J_host(L, 0.1), U_host(L, 1);
	memCopy(J, J_host.data(), L*sizeof(real), cudaMemcpyHostToDevice);
	memCopy(U, U_host.data(), L*sizeof(real), cudaMemcpyHostToDevice);

	real* f;
	memAlloc<real>(&f, 1);
	cudaMemset(f, 0, sizeof(real));
	real* g;
	memAlloc<real>(&g, 2 * L * dim);

	real* norm2s;
	memAlloc<real>(&norm2s, L);
//	vector<real> norm2s_host(L, 1);
	memCopy(norm2s, norm2_host.data(), L*sizeof(real), cudaMemcpyHostToDevice);

	printf("Before energy\n");
	energy(x, f, g, U, J, 0.5, norm2s);
	printf("After energy\n");

	memCopy(f_tb_host, f, sizeof(real), cudaMemcpyDeviceToHost);
	vector<real> g_host(ndim, 0);
	memCopy(g_host.data(), g, ndim*sizeof(real), cudaMemcpyDeviceToHost);

	real f1 = *f_tb_host;
	printf("f: %f\n", *f_tb_host);

	real dx = 1e-8;
	initProb(x, nbd, l, u, 6, dx);
	memCopy(x_host.data(), x, ndim*sizeof(real), cudaMemcpyDeviceToHost);
	for(int i = 0; i < L; i++) {
		norm2_host[i] = 0;
		for(int n = 0; n <= nmax; n++) {
			norm2_host[i] += x_host[2*(i*dim+n)]*x_host[2*(i*dim+n)]+x_host[2*(i*dim+n)+1]*x_host[2*(i*dim+n)+1];
		}
//		printf("norm2[%d] = %f\n", i, norm2_host[i]);
	}
	memCopy(norm2s, norm2_host.data(), L*sizeof(real), cudaMemcpyHostToDevice);
	cudaMemset(f, 0, sizeof(real));
//	energy(x, f, g, U, J, 0.5, norm2s);
	memCopy(f_tb_host, f, sizeof(real), cudaMemcpyDeviceToHost);

	real f2 = *f_tb_host;
	printf("f: %f\n", *f_tb_host);

//	printf("df: %e\n", (f1-f2)/dx);
//
//	printf("g: ");
//	for (int i = 0; i < ndim; i++) {
//		printf("%f, ", g_host[i]);
//	}
//	printf("\n");
//	real* x_host = new real[ndim];
	memCopy(x_host.data(), x, ndim * sizeof(real), cudaMemcpyDeviceToHost);
	printf("x: ");
	for (int i = 0; i < ndim; i++) {
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

	printf("Runtime: %ld", end - start);
}

