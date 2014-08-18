/*
 * phasediagram.cpp
 *
 *  Created on: Aug 12, 2014
 *      Author: Abuenameh
 */

#include <ctime>
#include <vector>
#include <algorithm>

#include "L-BFGS/lbfgsb.h"
#include "L-BFGS/cutil_inline.h"

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
real theta;
bool ct;

Estruct* Es;
Workspace work;
Parameters parms;

extern void initProb(int ndim, real* x, int* nbd, real* l, real* u);
extern void initProb(int ndim, real* x, int* nbd, real* l, real* u, int i,
	real dx);

void energy(real* x, real* f_dev, real* g_dev, Parameters& parms, real theta,
	Estruct* Es, Workspace& work, bool ct);

void funcgrad(real* x, real& f, real* g, const cudaStream_t& stream) {
	energy(x, f_tb_dev, g, parms, theta, Es, work, ct);
	f = *f_tb_host;
//	printf("f_tb_host: %f\n", *f_tb_host);
}

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

	allocE(&Es);
	CudaCheckError();
	allocWorkspace(work);
	CudaCheckError();

	real* x;
	int* nbd;
	real* l;
	real* u;
	CudaSafeMemAllocCall(memAlloc<real>(&x, ndim));
	CudaSafeMemAllocCall(memAlloc<int>(&nbd, ndim));
	CudaSafeMemAllocCall(memAlloc<real>(&l, ndim));
	CudaSafeMemAllocCall(memAlloc<real>(&u, ndim));
	CudaSafeMemAllocCall(
		memAllocHost<real>(&f_tb_host, &f_tb_dev, sizeof(real)));

	vector<real> x_host(ndim);
	vector<real> norm2_host(L, 0);

	U = 1;
	J = 0.2;
	mu = 0.5;

//	for(int i = 0; i < 10; i++) {

	printf("Before initProb\n");
	initProb(ndim, x, nbd, l, u);
	memCopy(x_host.data(), x, ndim * sizeof(real), cudaMemcpyDeviceToHost);
	CudaCheckError();

//	printf("f: %.20e\n", *f_tb_host);
//	}

	real* U;
	real* J;
	CudaSafeMemAllocCall(memAlloc<real>(&J, L));
	CudaSafeMemAllocCall(memAlloc<real>(&U, L));
	vector<real> J_host(L, 0.1), U_host(L, 1);
	std::generate(J_host.begin(), J_host.end(), std::rand);
	std::generate(U_host.begin(), U_host.end(), std::rand);
	for (int i = 0; i < L; i++) {
		J_host[i] = 0.1;//0.00001*(i+1)*(i+2);
		U_host[i] = 1;//0.0000002*(i+1)*(2*i+1)*(3*i*i+2);
//		J_host[i] /= RAND_MAX;
//		J_host[i] /= 10;
//		U_host[i] /= RAND_MAX;
//		printf("%f %f\n", J_host[i], U_host[i]);
	}
	memCopy(J, J_host.data(), L * sizeof(real), cudaMemcpyHostToDevice);
	CudaCheckError();
	memCopy(U, U_host.data(), L * sizeof(real), cudaMemcpyHostToDevice);
	CudaCheckError();

	parms.J = J;
	parms.U = U;
	parms.mu = mu;

	theta = 0;

//	real* g;
//	memAlloc<real>(&g, ndim);
//	initProb(ndim, x, nbd, l, u);
//	energy(x, f_tb_host, g, parms, theta, Es, work, false);
//	real f1 = *f_tb_host;
//	vector<real> g_host(ndim);
//	memCopy(g_host.data(), g, ndim*sizeof(real), cudaMemcpyDeviceToHost);
//	printf("g: ");
//	for(int i = 0; i < ndim; i++) {
//		printf("%f, ", g_host[i]);
//	}
//	printf("\n");
//	initProb(ndim, x, nbd, l, u, 6, 1e-7);
//	energy(x, f_tb_host, g, parms, theta, Es, work, false);
//	real f2 = *f_tb_host;
//	printf("f2-f1 = %e\n", f2-f1);

	lbfgsbminimize(ndim, 4, x, epsg, epsf, epsx, maxits, nbd, l, u, info);
	printf("info: %d\n", info);

	real E0 = *f_tb_host;
	printf("E0: %f\n", *f_tb_host);

	memCopy(x_host.data(), x, ndim * sizeof(real), cudaMemcpyDeviceToHost);
	printf("f0: ");
	for (int i = 0; i < ndim; i++) {
		printf("%f, ", x_host[i]);
	}
	printf("\n");

	theta = 0.01;

	lbfgsbminimize(ndim, 4, x, epsg, epsf, epsx, maxits, nbd, l, u, info);
	printf("info: %d\n", info);

	real Eth = *f_tb_host;
	printf("Eth: %f\n", *f_tb_host);

	memCopy(x_host.data(), x, ndim * sizeof(real), cudaMemcpyDeviceToHost);
	printf("fth: ");
	for (int i = 0; i < ndim; i++) {
		printf("%f, ", x_host[i]);
	}
	printf("\n");

	real Jmean = 0;
	for(int i = 0; i < L; i++) {
		Jmean += J_host[i];
	}
	Jmean /= L;
	printf("fs?: %f\n", (Eth-E0)/(Jmean*theta*theta));

//	real* f;
//	CudaSafeMemAllocCall(memAlloc<real>(&f, 1));
//	cudaMemset(f, 0, sizeof(real));
//	real* g;
//	CudaSafeMemAllocCall(memAlloc<real>(&g, 2 * L * dim));

//	printf("Before energy\n");
//	energy(x, f, g, parms, Es, work);
//	CudaCheckError();
//	printf("After energy\n");

//	memCopy(f_tb_host, f, sizeof(real), cudaMemcpyDeviceToHost);
//	vector<real> g_host(ndim, 0);
//	memCopy(g_host.data(), g, ndim * sizeof(real), cudaMemcpyDeviceToHost);

//	real f1 = *f_tb_host;
//	printf("f: %f\n", *f_tb_host);

//	real dx = 1e-8;
//	initProb(x, nbd, l, u, 6, dx);
//	memCopy(x_host.data(), x, ndim*sizeof(real), cudaMemcpyDeviceToHost);
//	cudaMemset(f, 0, sizeof(real));
//	energy(x, f, g, U, J, 0.5, norm2s);
//	memCopy(f_tb_host, f, sizeof(real), cudaMemcpyDeviceToHost);

//	real f2 = *f_tb_host;
//	printf("f: %f\n", *f_tb_host);

//	memCopy(x_host.data(), x, ndim * sizeof(real), cudaMemcpyDeviceToHost);
//	printf("x: ");
//	for (int i = 0; i < ndim; i++) {
//		printf("%f, ", x_host[i]);
//	}
//	printf("\n");

	memFreeHost(f_tb_host);
	memFree(x);
	memFree(nbd);
	memFree(l);
	memFree(u);

	freeE(Es);
	freeWorkspace(work);

	cublasDestroy_v2(cublasHd);

	cudaDeviceReset();

	time_t end = time(NULL);

	printf("Runtime: %ld", end - start);
}

