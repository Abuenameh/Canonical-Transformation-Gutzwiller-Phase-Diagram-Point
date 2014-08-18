/*
 * functions.cu
 *
 *  Created on: Aug 12, 2014
 *      Author: Abuenameh
 */

#include <stdio.h>

#include "L-BFGS/cutil_inline.h"
#include "L-BFGS/lbfgsbcuda.h"
#include "L-BFGS/lbfgsb.h"
#include "cudacomplex.hpp"
#include "phasediagram.hpp"

__device__ double atomicAdd(double* address, double val) {
	unsigned long long int* address_as_ull = (unsigned long long int*) address;
	unsigned long long int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
			__double_as_longlong(val + __longlong_as_double(assumed)));
	} while (assumed != old);
	return __longlong_as_double(old);
}

__device__ void atomicAdd(doublecomplex* address, doublecomplex val) {
	atomicAdd(&address->real(), val.real());
	atomicAdd(&address->imag(), val.imag());
}

__global__ void initProbKer(int ndim, real* x, int* nbd, real* l, real* u) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= ndim) {
		return;
	}

//	x[i] = 1 / sqrt(2.0 * dim);
	real scale = 1;
	int k = i / (2 * dim);
	int n = i % (2 * dim);
//	if (k == 0)
//		scale = sqrt(35 / 2.);
//	if (k == 1)
//		scale = sqrt(51 / 2.);
//	if (k == 2)
//		scale = sqrt(71 / 2.);
	x[i] = (1 / sqrt(2.0 * dim)); // * (k + n) / scale;
	nbd[i] = 0;
//	l[i] = -100;
//	u[i] = 100;
}

__global__ void initProbKer(int ndim, real* x, int* nbd, real* l, real* u,
	int j, real dx) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= ndim) {
		return;
	}

	int k = i / (2 * dim);
	int n = i % (2 * dim);
	x[i] = (1 / sqrt(2.0 * dim)) * (k + n);
	nbd[i] = 0;
	if (i == j)
		x[j] += dx;
}

extern void initProb(int ndim, real* x, int* nbd, real* l, real* u) {
	initProbKer<<<lbfgsbcuda::iDivUp(ndim, 64), 64>>>(ndim, x, nbd, l, u);
//	initProbKer<<<1,2*L*dim>>>(x, nbd, l, u);
}

extern void initProb(int ndim, real* x, int* nbd, real* l, real* u, int i,
	real dx) {
	initProbKer<<<lbfgsbcuda::iDivUp(ndim, 64), 64>>>(ndim, x, nbd, l, u, i,
		dx);
}

__global__ void energyfctKer(real* x, real* norm2, Parameters parms, real theta,
	doublecomplex* E, Estruct* Es) {
	int n = blockIdx.x;
	int i = threadIdx.x;

	if (n > nmax || i >= L) {
		return;
	}

	int k1 = mod(i - 2);
	int j1 = mod(i - 1);
	int j2 = mod(i + 1);
	int k2 = mod(i + 2);

	__shared__ doublecomplex fi[L * dim];
	__shared__ doublecomplex* f[L];
	__shared__ real U[L];
	__shared__ real J[L];
	__shared__ real mu;
	__shared__ real costh, cos2th;
	if (i == 0) {
		for (int j = 0; j < L; j++) {
			int k = j * dim;
			f[j] = &fi[k];
			for (int m = 0; m <= nmax; m++) {
				f[j][m] = make_doublecomplex(x[2 * (k + m)],
					x[2 * (k + m) + 1]);
			}
			U[j] = parms.U[j];
			J[j] = parms.J[j];
		}
		mu = parms.mu;
		costh = cos(theta);
		cos2th = cos(2 * theta);
	}
	__syncthreads();

	doublecomplex E0 = doublecomplex::zero();
	doublecomplex E1j1 = doublecomplex::zero();
	doublecomplex E1j2 = doublecomplex::zero();
	doublecomplex E2j1 = doublecomplex::zero();
	doublecomplex E2j2 = doublecomplex::zero();
	doublecomplex E3j1 = doublecomplex::zero();
	doublecomplex E3j2 = doublecomplex::zero();
	doublecomplex E4j1j2 = doublecomplex::zero();
	doublecomplex E4j1k1 = doublecomplex::zero();
	doublecomplex E4j2k2 = doublecomplex::zero();
	doublecomplex E5j1j2 = doublecomplex::zero();
	doublecomplex E5j1k1 = doublecomplex::zero();
	doublecomplex E5j2k2 = doublecomplex::zero();

	E0 = (0.5 * U[i] * n * (n - 1) - mu * n) * ~f[i][n] * f[i][n];

	if (n < nmax) {
		E1j1 += -J[j1] * costh * g(n, n + 1) * ~f[i][n + 1] * ~f[j1][n]
			* f[i][n] * f[j1][n + 1];
		E1j2 += -J[i] * costh * g(n, n + 1) * ~f[i][n + 1] * ~f[j2][n] * f[i][n]
			* f[j2][n + 1];

		if (n > 0) {
			E2j1 += 0.5 * J[j1] * J[j1] * cos2th * g(n, n) * g(n - 1, n + 1)
				* ~f[i][n + 1] * ~f[j1][n - 1] * f[i][n - 1] * f[j1][n + 1]
				* (1 / eps(U, i, j1, n, n) - 1 / eps(U, i, j1, n - 1, n + 1));
			E2j2 += 0.5 * J[i] * J[i] * cos2th * g(n, n) * g(n - 1, n + 1)
				* ~f[i][n + 1] * ~f[j2][n - 1] * f[i][n - 1] * f[j2][n + 1]
				* (1 / eps(U, i, j2, n, n) - 1 / eps(U, i, j2, n - 1, n + 1));
		}

		for (int m = 1; m <= nmax; m++) {
			if (n != m - 1) {
				E3j1 += 0.5 * (J[j1] * J[j1] / eps(U, i, j1, n, m)) * g(n, m)
					* g(m - 1, n + 1)
					* (~f[i][n + 1] * ~f[j1][m - 1] * f[i][n + 1] * f[j1][m - 1]
						- ~f[i][n] * ~f[j1][m] * f[i][n] * f[j1][m]);
				E3j2 += 0.5 * (J[i] * J[i] / eps(U, i, j2, n, m)) * g(n, m)
					* g(m - 1, n + 1)
					* (~f[i][n + 1] * ~f[j2][m - 1] * f[i][n + 1] * f[j2][m - 1]
						- ~f[i][n] * ~f[j2][m] * f[i][n] * f[j2][m]);
			}
		}

		if (n > 0) {
			E4j1j2 += 0.5 * (J[j1] * J[i] / eps(U, i, j1, n, n)) * g(n, n)
				* g(n - 1, n + 1) * ~f[i][n + 1] * ~f[j1][n - 1] * ~f[j2][n]
				* f[i][n - 1] * f[j1][n] * f[j2][n + 1];
			E4j1j2 += 0.5 * (J[i] * J[j1] / eps(U, i, j2, n, n)) * g(n, n)
				* g(n - 1, n + 1) * ~f[i][n + 1] * ~f[j2][n - 1] * ~f[j1][n]
				* f[i][n - 1] * f[j2][n] * f[j1][n + 1];
			E4j1k1 += 0.5 * (J[j1] * J[k1] / eps(U, i, j1, n, n)) * g(n, n)
				* g(n - 1, n + 1) * ~f[i][n + 1] * ~f[j1][n - 1] * ~f[k1][n]
				* f[i][n] * f[j1][n + 1] * f[k1][n - 1];
			E4j2k2 += 0.5 * (J[i] * J[j2] / eps(U, i, j2, n, n)) * g(n, n)
				* g(n - 1, n + 1) * ~f[i][n + 1] * ~f[j2][n - 1] * ~f[k2][n]
				* f[i][n] * f[j2][n + 1] * f[k2][n - 1];
			E4j1j2 -= 0.5 * (J[j1] * J[i] / eps(U, i, j1, n - 1, n + 1))
				* g(n, n) * g(n - 1, n + 1) * ~f[i][n + 1] * ~f[j1][n]
				* ~f[j2][n - 1] * f[i][n - 1] * f[j1][n + 1] * f[j2][n];
			E4j1j2 -= 0.5 * (J[i] * J[j1] / eps(U, i, j2, n - 1, n + 1))
				* g(n, n) * g(n - 1, n + 1) * ~f[i][n + 1] * ~f[j2][n]
				* ~f[j1][n - 1] * f[i][n - 1] * f[j2][n + 1] * f[j1][n];
			E4j1k1 -= 0.5 * (J[j1] * J[k1] / eps(U, i, j1, n - 1, n + 1))
				* g(n, n) * g(n - 1, n + 1) * ~f[i][n] * ~f[j1][n - 1]
				* ~f[k1][n + 1] * f[i][n - 1] * f[j1][n + 1] * f[k1][n];
			E4j2k2 -= 0.5 * (J[i] * J[j2] / eps(U, i, j2, n - 1, n + 1))
				* g(n, n) * g(n - 1, n + 1) * ~f[i][n] * ~f[j2][n - 1]
				* ~f[k2][n + 1] * f[i][n - 1] * f[j2][n + 1] * f[k2][n];
		}

		for (int m = 1; m <= nmax; m++) {
			if (n != m - 1 && n < nmax) {
				E5j1j2 += 0.5 * (J[j1] * J[i] * cos2th / eps(U, i, j1, n, m))
					* g(n, m) * g(m - 1, n + 1) * ~f[i][n + 1] * ~f[j1][m - 1]
					* ~f[j2][m] * f[i][n + 1] * f[j1][m] * f[j2][m - 1];
				E5j1j2 += 0.5 * (J[i] * J[j1] * cos2th / eps(U, i, j2, n, m))
					* g(n, m) * g(m - 1, n + 1) * ~f[i][n + 1] * ~f[j2][m - 1]
					* ~f[j1][m] * f[i][n + 1] * f[j2][m] * f[j1][m - 1];
				E5j1k1 += 0.5 * (J[j1] * J[k1] * cos2th / eps(U, i, j1, n, m))
					* g(n, m) * g(m - 1, n + 1) * ~f[i][n + 1] * ~f[j1][m - 1]
					* ~f[k1][n] * f[i][n] * f[j1][m - 1] * f[k1][n + 1];
				E5j2k2 += 0.5 * (J[i] * J[j2] * cos2th / eps(U, i, j2, n, m))
					* g(n, m) * g(m - 1, n + 1) * ~f[i][n + 1] * ~f[j2][m - 1]
					* ~f[k2][n] * f[i][n] * f[j2][m - 1] * f[k2][n + 1];
				E5j1j2 -= 0.5 * (J[j1] * J[i] * cos2th / eps(U, i, j1, n, m))
					* g(n, m) * g(m - 1, n + 1) * ~f[i][n] * ~f[j1][m - 1]
					* ~f[j2][m] * f[i][n] * f[j1][m] * f[j2][m - 1];
				E5j1j2 -= 0.5 * (J[i] * J[j1] * cos2th / eps(U, i, j2, n, m))
					* g(n, m) * g(m - 1, n + 1) * ~f[i][n] * ~f[j2][m - 1]
					* ~f[j1][m] * f[i][n] * f[j2][m] * f[j1][m - 1];
				E5j1k1 -= 0.5 * (J[j1] * J[k1] * cos2th / eps(U, i, j1, n, m))
					* g(n, m) * g(m - 1, n + 1) * ~f[i][n + 1] * ~f[j1][m]
					* ~f[k1][n] * f[i][n] * f[j1][m] * f[k1][n + 1];
				E5j2k2 -= 0.5 * (J[i] * J[j2] * cos2th / eps(U, i, j2, n, m))
					* g(n, m) * g(m - 1, n + 1) * ~f[i][n + 1] * ~f[j2][m]
					* ~f[k2][n] * f[i][n] * f[j2][m] * f[k2][n + 1];
			}
		}
	}

	atomicAdd(E, E0 / norm2[i]);
	atomicAdd(E, E1j1 / (norm2[i] * norm2[j1]));
	atomicAdd(E, E1j2 / (norm2[i] * norm2[j2]));
	atomicAdd(E, E2j1 / (norm2[i] * norm2[j1]));
	atomicAdd(E, E2j2 / (norm2[i] * norm2[j2]));
	atomicAdd(E, E3j1 / (norm2[i] * norm2[j1]));
	atomicAdd(E, E3j2 / (norm2[i] * norm2[j2]));
	atomicAdd(E, E4j1j2 / (norm2[i] * norm2[j1] * norm2[j2]));
	atomicAdd(E, E4j1k1 / (norm2[i] * norm2[j1] * norm2[k1]));
	atomicAdd(E, E4j2k2 / (norm2[i] * norm2[j2] * norm2[k2]));
	atomicAdd(E, E5j1j2 / (norm2[i] * norm2[j1] * norm2[j2]));
	atomicAdd(E, E5j1k1 / (norm2[i] * norm2[j1] * norm2[k1]));
	atomicAdd(E, E5j2k2 / (norm2[i] * norm2[j2] * norm2[k2]));

	atomicAdd(&Es->E0[i], E0);
	atomicAdd(&Es->E1j1[i], E1j1);
	atomicAdd(&Es->E1j2[i], E1j2);
	atomicAdd(&Es->E2j1[i], E2j1);
	atomicAdd(&Es->E2j2[i], E2j2);
	atomicAdd(&Es->E3j1[i], E3j1);
	atomicAdd(&Es->E3j2[i], E3j2);
	atomicAdd(&Es->E4j1j2[i], E4j1j2);
	atomicAdd(&Es->E4j1k1[i], E4j1k1);
	atomicAdd(&Es->E4j2k2[i], E4j2k2);
	atomicAdd(&Es->E5j1j2[i], E5j1j2);
	atomicAdd(&Es->E5j1k1[i], E5j1k1);
	atomicAdd(&Es->E5j2k2[i], E5j2k2);

//	printf("%d: %f, %f, %f, %f, %f, %f, %f\n", i, E0.real(), E1j1.real(), E1j2.real(), E2j1.real(), E2j2.real(), E3j1.real(), E3j2.real());
//	printf("%d: %f, %f, %f, %f, %f, %f\n", i, E4j1j2.real(), E4j1k1.real(), E4j2k2.real(), E5j1j2.real(), E5j1k1.real(), E5j2k2.real());
}

__global__ void energygctKer(real* x, real* norm2, Parameters parms, real theta, Estruct* Es,
	real* g_dev) {
	int n = blockIdx.x;
	int i = threadIdx.x;

	if (n > nmax || i >= L) {
		return;
	}

	int k1 = mod(i - 2);
	int j1 = mod(i - 1);
	int j2 = mod(i + 1);
	int k2 = mod(i + 2);

	__shared__ doublecomplex fi[L * dim];
	__shared__ doublecomplex* f[L];
	__shared__ real U[L];
	__shared__ real J[L];
	__shared__ real mu;
	__shared__ real costh, cos2th;
	__shared__ doublecomplex E0[L];
	__shared__ doublecomplex E1j1[L];
	__shared__ doublecomplex E1j2[L];
	__shared__ doublecomplex E2j1[L];
	__shared__ doublecomplex E2j2[L];
	__shared__ doublecomplex E3j1[L];
	__shared__ doublecomplex E3j2[L];
	__shared__ doublecomplex E4j1j2[L];
	__shared__ doublecomplex E4j1k1[L];
	__shared__ doublecomplex E4j2k2[L];
	__shared__ doublecomplex E5j1j2[L];
	__shared__ doublecomplex E5j1k1[L];
	__shared__ doublecomplex E5j2k2[L];
	if (i == 0) {
		for (int j = 0; j < L; j++) {
			int k = j * dim;
			f[j] = &fi[k];
			for (int m = 0; m <= nmax; m++) {
				f[j][m] = make_doublecomplex(x[2 * (k + m)],
					x[2 * (k + m) + 1]);
			}
			U[j] = parms.U[j];
			J[j] = parms.J[j];
			costh = cos(theta);
			cos2th = cos(2 * theta);
			E0[j] = Es->E0[j];
			E1j1[j] = Es->E1j1[j];
			E1j2[j] = Es->E1j2[j];
			E2j1[j] = Es->E2j1[j];
			E2j2[j] = Es->E2j2[j];
			E3j1[j] = Es->E3j1[j];
			E3j2[j] = Es->E3j2[j];
			E4j1j2[j] = Es->E4j1j2[j];
			E4j1k1[j] = Es->E4j1k1[j];
			E4j2k2[j] = Es->E4j2k2[j];
			E5j1j2[j] = Es->E5j1j2[j];
			E5j1k1[j] = Es->E5j1k1[j];
			E5j2k2[j] = Es->E5j2k2[j];
		}
		mu = parms.mu;
	}
	__syncthreads();

	doublecomplex E0df = doublecomplex::zero();
	doublecomplex E1j1df = doublecomplex::zero();
	doublecomplex E1j2df = doublecomplex::zero();
	doublecomplex E2j1df = doublecomplex::zero();
	doublecomplex E2j2df = doublecomplex::zero();
	doublecomplex E3j1df = doublecomplex::zero();
	doublecomplex E3j2df = doublecomplex::zero();
	doublecomplex E4j1j2df = doublecomplex::zero();
	doublecomplex E4j1k1df = doublecomplex::zero();
	doublecomplex E4j2k2df = doublecomplex::zero();
	doublecomplex E5j1j2df = doublecomplex::zero();
	doublecomplex E5j1k1df = doublecomplex::zero();
	doublecomplex E5j2k2df = doublecomplex::zero();

	E0df = (0.5 * U[i] * n * (n - 1) - mu * n) * f[i][n];

	if (n < nmax) {
		E1j1df += -J[j1] * costh * g(n, n + 1) * ~f[j1][n + 1] * f[j1][n]
			* f[i][n + 1];
		E1j2df += -J[i] * costh * g(n, n + 1) * ~f[j2][n + 1] * f[j2][n]
			* f[i][n + 1];
	}
	if (n > 0) {
		E1j1df += -J[j1] * costh * g(n - 1, n) * ~f[j1][n - 1] * f[j1][n]
			* f[i][n - 1];
		E1j2df += -J[i] * costh * g(n - 1, n) * ~f[j2][n - 1] * f[j2][n]
			* f[i][n - 1];
	}

	if (n > 1) {
		E2j1df += 0.5 * J[j1] * J[j1] * cos2th * g(n - 1, n - 1) * g(n - 2, n)
			* ~f[j1][n - 2] * f[j1][n] * f[i][n - 2]
			* (1 / eps(U, i, j1, n - 1, n - 1) - 1 / eps(U, i, j1, n - 2, n));
		E2j2df += 0.5 * J[i] * J[i] * cos2th * g(n - 1, n - 1) * g(n - 2, n)
			* ~f[j2][n - 2] * f[j2][n] * f[i][n - 2]
			* (1 / eps(U, i, j2, n - 1, n - 1) - 1 / eps(U, i, j2, n - 2, n));
	}
	if (n < nmax - 1) {
		E2j1df += 0.5 * J[j1] * J[j1] * cos2th * g(n + 1, n + 1) * g(n, n + 2)
			* ~f[j1][n + 2] * f[j1][n] * f[i][n + 2]
			* (1 / eps(U, j1, i, n + 1, n + 1) - 1 / eps(U, j1, i, n, n + 2));
		E2j2df += 0.5 * J[i] * J[i] * cos2th * g(n + 1, n + 1) * g(n, n + 2)
			* ~f[j2][n + 2] * f[j2][n] * f[i][n + 2]
			* (1 / eps(U, j2, i, n + 1, n + 1) - 1 / eps(U, j2, i, n, n + 2));
	}

	for (int m = 0; m < nmax; m++) {
		if (n != m + 1) {
			E3j1df += 0.5 * (J[j1] * J[j1] / eps(U, i, j1, n - 1, m + 1))
				* g(n - 1, m + 1) * g(m, n) * ~f[j1][m] * f[j1][m] * f[i][n];
			E3j2df += 0.5 * (J[i] * J[i] / eps(U, i, j2, n - 1, m + 1))
				* g(n - 1, m + 1) * g(m, n) * ~f[j2][m] * f[j2][m] * f[i][n];
			E3j1df -= 0.5 * (J[j1] * J[j1] / eps(U, j1, i, m, n)) * g(m, n)
				* g(n - 1, m + 1) * ~f[j1][m] * f[j1][m] * f[i][n];
			E3j2df -= 0.5 * (J[i] * J[i] / eps(U, j2, i, m, n)) * g(m, n)
				* g(n - 1, m + 1) * ~f[j2][m] * f[j2][m] * f[i][n];
		}
		if (n != m && n < nmax) {
			E3j1df += 0.5 * (J[j1] * J[j1] / eps(U, j1, i, m, n + 1))
				* g(m, n + 1) * g(n, m + 1) * ~f[j1][m + 1] * f[j1][m + 1]
				* f[i][n];
			E3j2df += 0.5 * (J[i] * J[i] / eps(U, j2, i, m, n + 1))
				* g(m, n + 1) * g(n, m + 1) * ~f[j2][m + 1] * f[j2][m + 1]
				* f[i][n];
			E3j1df -= 0.5 * (J[j1] * J[j1] / eps(U, i, j1, n, m + 1))
				* g(n, m + 1) * g(m, n + 1) * ~f[j1][m + 1] * f[j1][m + 1]
				* f[i][n];
			E3j2df -= 0.5 * (J[i] * J[i] / eps(U, i, j2, n, m + 1))
				* g(n, m + 1) * g(m, n + 1) * ~f[j2][m + 1] * f[j2][m + 1]
				* f[i][n];
		}
	}

	if (n >= 2) {
		E4j1j2df += 0.5 * g(n - 1, n - 1) * g(n - 2, n)
			* (J[j1] * J[i] / eps(U, i, j1, n - 1, n - 1)) * ~f[j1][n - 2]
			* ~f[j2][n - 1] * f[j1][n - 1] * f[j2][n] * f[i][n - 2];
		E4j1j2df += 0.5 * g(n - 1, n - 1) * g(n - 2, n)
			* (J[i] * J[j1] / eps(U, i, j2, n - 1, n - 1)) * ~f[j2][n - 2]
			* ~f[j1][n - 1] * f[j2][n - 1] * f[j1][n] * f[i][n - 2];
		E4j1k1df += 0.5 * g(n - 1, n - 1) * g(n - 2, n)
			* (J[j1] * J[k1] / eps(U, i, j1, n - 1, n - 1)) * ~f[j1][n - 2]
			* ~f[k1][n - 1] * f[j1][n] * f[k1][n - 2] * f[i][n - 1];
		E4j2k2df += 0.5 * g(n - 1, n - 1) * g(n - 2, n)
			* (J[i] * J[j2] / eps(U, i, j2, n - 1, n - 1)) * ~f[j2][n - 2]
			* ~f[k2][n - 1] * f[j2][n] * f[k2][n - 2] * f[i][n - 1];
		E4j1j2df -= 0.5 * g(n - 1, n - 1) * g(n - 2, n)
			* (J[j1] * J[i] / eps(U, i, j1, n - 2, n)) * ~f[j1][n - 1]
			* ~f[j2][n - 2] * f[j1][n] * f[j2][n - 1] * f[i][n - 2];
		E4j1j2df -= 0.5 * g(n - 1, n - 1) * g(n - 2, n)
			* (J[i] * J[j1] / eps(U, i, j2, n - 2, n)) * ~f[j2][n - 1]
			* ~f[j1][n - 2] * f[j2][n] * f[j1][n - 1] * f[i][n - 2];
		E4j1k1df -= 0.5 * g(n - 1, n - 1) * g(n - 2, n)
			* (J[k1] * J[j1] / eps(U, k1, j1, n - 2, n)) * ~f[k1][n - 1]
			* ~f[j1][n - 2] * f[k1][n - 2] * f[j1][n] * f[i][n - 1];
		E4j2k2df -= 0.5 * g(n - 1, n - 1) * g(n - 2, n)
			* (J[j2] * J[i] / eps(U, k2, j2, n - 2, n)) * ~f[k2][n - 1]
			* ~f[j2][n - 2] * f[k2][n - 2] * f[j2][n] * f[i][n - 1];
	}
	if (n >= 1 && n <= nmax - 1) {
		E4j1k1df += 0.5 * g(n, n) * g(n - 1, n + 1)
			* (J[k1] * J[j1] / eps(U, j1, k1, n, n)) * ~f[j1][n + 1]
			* ~f[k1][n - 1] * f[j1][n - 1] * f[k1][n] * f[i][n + 1];
		E4j2k2df += 0.5 * g(n, n) * g(n - 1, n + 1)
			* (J[j2] * J[i] / eps(U, j2, k2, n, n)) * ~f[j2][n + 1]
			* ~f[k2][n - 1] * f[j2][n - 1] * f[k2][n] * f[i][n + 1];
		E4j1k1df += 0.5 * g(n, n) * g(n - 1, n + 1)
			* (J[k1] * J[j1] / eps(U, k1, j1, n, n)) * ~f[k1][n + 1]
			* ~f[j1][n - 1] * f[k1][n] * f[j1][n + 1] * f[i][n - 1];
		E4j2k2df += 0.5 * g(n, n) * g(n - 1, n + 1)
			* (J[j2] * J[i] / eps(U, k2, j2, n, n)) * ~f[k2][n + 1]
			* ~f[j2][n - 1] * f[k2][n] * f[j2][n + 1] * f[i][n - 1];
		E4j1k1df -= 0.5 * g(n, n) * g(n - 1, n + 1)
			* (J[j1] * J[k1] / eps(U, j1, i, n - 1, n + 1)) * ~f[j1][n + 1]
			* ~f[k1][n - 1] * f[j1][n - 1] * f[k1][n] * f[i][n + 1];
		E4j2k2df -= 0.5 * g(n, n) * g(n - 1, n + 1)
			* (J[i] * J[j2] / eps(U, j2, i, n - 1, n + 1)) * ~f[j2][n + 1]
			* ~f[k2][n - 1] * f[j2][n - 1] * f[k2][n] * f[i][n + 1];
		E4j1k1df -= 0.5 * g(n, n) * g(n - 1, n + 1)
			* (J[j1] * J[k1] / eps(U, i, j1, n - 1, n + 1)) * ~f[j1][n - 1]
			* ~f[k1][n + 1] * f[j1][n + 1] * f[k1][n] * f[i][n - 1];
		E4j2k2df -= 0.5 * g(n, n) * g(n - 1, n + 1)
			* (J[i] * J[j2] / eps(U, i, j2, n - 1, n + 1)) * ~f[j2][n - 1]
			* ~f[k2][n + 1] * f[j2][n + 1] * f[k2][n] * f[i][n - 1];
	}
	if (n <= nmax - 2) {
		E4j1k1df += 0.5 * g(n + 1, n + 1) * g(n, n + 2)
			* (J[j1] * J[k1] / eps(U, j1, i, n + 1, n + 1)) * ~f[j1][n + 2]
			* ~f[k1][n + 1] * f[j1][n] * f[k1][n + 2] * f[i][n + 1];
		E4j2k2df += 0.5 * g(n + 1, n + 1) * g(n, n + 2)
			* (J[i] * J[j2] / eps(U, j2, i, n + 1, n + 1)) * ~f[j2][n + 2]
			* ~f[k2][n + 1] * f[j2][n] * f[k2][n + 2] * f[i][n + 1];
		E4j1j2df += 0.5 * g(n + 1, n + 1) * g(n, n + 2)
			* (J[j1] * J[i] / eps(U, j1, i, n + 1, n + 1)) * ~f[j1][n + 2]
			* ~f[j2][n + 1] * f[j1][n + 1] * f[j2][n] * f[i][n + 2];
		E4j1j2df += 0.5 * g(n + 1, n + 1) * g(n, n + 2)
			* (J[i] * J[j1] / eps(U, j2, i, n + 1, n + 1)) * ~f[j2][n + 2]
			* ~f[j1][n + 1] * f[j2][n + 1] * f[j1][n] * f[i][n + 2];
		E4j1k1df -= 0.5 * g(n + 1, n + 1) * g(n, n + 2)
			* (J[k1] * J[j1] / eps(U, j1, k1, n, n + 2)) * ~f[j1][n + 2]
			* ~f[k1][n + 1] * f[j1][n] * f[k1][n + 2] * f[i][n + 1];
		E4j2k2df -= 0.5 * g(n + 1, n + 1) * g(n, n + 2)
			* (J[j2] * J[i] / eps(U, j2, k2, n, n + 2)) * ~f[j2][n + 2]
			* ~f[k2][n + 1] * f[j2][n] * f[k2][n + 2] * f[i][n + 1];
		E4j1j2df -= 0.5 * g(n + 1, n + 1) * g(n, n + 2)
			* (J[j1] * J[i] / eps(U, j1, i, n, n + 2)) * ~f[j1][n + 1]
			* ~f[j2][n + 2] * f[j1][n] * f[j2][n + 1] * f[i][n + 2];
		E4j1j2df -= 0.5 * g(n + 1, n + 1) * g(n, n + 2)
			* (J[i] * J[j1] / eps(U, j2, i, n, n + 2)) * ~f[j2][n + 1]
			* ~f[j1][n + 2] * f[j2][n] * f[j1][n + 1] * f[i][n + 2];
	}

	for (int m = 0; m <= nmax; m++) {
		if (n != m) {
			if (n > 0) {
				if (m > 0) {
					E5j1j2df += 0.5
						* (J[j1] * J[i] * cos2th / eps(U, i, j1, n - 1, m))
						* g(n - 1, m) * g(m - 1, n) * ~f[j1][m - 1] * ~f[j2][m]
						* f[j1][m] * f[j2][m - 1] * f[i][n];
					E5j1j2df += 0.5
						* (J[i] * J[j1] * cos2th / eps(U, i, j2, n - 1, m))
						* g(n - 1, m) * g(m - 1, n) * ~f[j2][m - 1] * ~f[j1][m]
						* f[j2][m] * f[j1][m - 1] * f[i][n];
					E5j1k1df += 0.5
						* (J[j1] * J[k1] * cos2th / eps(U, i, j1, n - 1, m))
						* g(n - 1, m) * g(m - 1, n) * ~f[j1][m - 1]
						* ~f[k1][n - 1] * f[j1][m - 1] * f[k1][n] * f[i][n - 1];
					E5j2k2df += 0.5
						* (J[i] * J[j2] * cos2th / eps(U, i, j2, n - 1, m))
						* g(n - 1, m) * g(m - 1, n) * ~f[j2][m - 1]
						* ~f[k2][n - 1] * f[j2][m - 1] * f[k2][n] * f[i][n - 1];
					E5j1k1df -= 0.5
						* (J[j1] * J[k1] * cos2th / eps(U, i, j1, n - 1, m))
						* g(n - 1, m) * g(m - 1, n) * ~f[j1][m] * ~f[k1][n - 1]
						* f[j1][m] * f[k1][n] * f[i][n - 1];
					E5j2k2df -= 0.5
						* (J[i] * J[j2] * cos2th / eps(U, i, j2, n - 1, m))
						* g(n - 1, m) * g(m - 1, n) * ~f[j2][m] * ~f[k2][n - 1]
						* f[j2][m] * f[k2][n] * f[i][n - 1];
				}
			}
			if (n < nmax) {
				if (m < nmax) {
					E5j1k1df += 0.5
						* (J[j1] * J[k1] * cos2th / eps(U, j1, i, m, n + 1))
						* g(m, n + 1) * g(n, m + 1) * ~f[j1][m + 1]
						* ~f[k1][n + 1] * f[j1][m + 1] * f[k1][n] * f[i][n + 1];
					E5j2k2df += 0.5
						* (J[i] * J[j2] * cos2th / eps(U, j2, i, m, n + 1))
						* g(m, n + 1) * g(n, m + 1) * ~f[j2][m + 1]
						* ~f[k2][n + 1] * f[j2][m + 1] * f[k2][n] * f[i][n + 1];
					E5j1j2df += 0.5
						* (J[j1] * J[i] * cos2th / eps(U, j1, i, m, n + 1))
						* g(m, n + 1) * g(n, m + 1) * ~f[j1][m + 1] * ~f[j2][m]
						* f[j1][m] * f[j2][m + 1] * f[i][n];
					E5j1j2df += 0.5
						* (J[i] * J[j1] * cos2th / eps(U, j2, i, m, n + 1))
						* g(m, n + 1) * g(n, m + 1) * ~f[j2][m + 1] * ~f[j1][m]
						* f[j2][m] * f[j1][m + 1] * f[i][n];
					E5j1k1df -= 0.5
						* (J[j1] * J[k1] * cos2th / eps(U, j1, i, m, n + 1))
						* g(m, n + 1) * g(n, m + 1) * ~f[j1][m] * ~f[k1][n + 1]
						* f[j1][m] * f[k1][n] * f[i][n + 1];
					E5j2k2df -= 0.5
						* (J[i] * J[j2] * cos2th / eps(U, j2, i, m, n + 1))
						* g(m, n + 1) * g(n, m + 1) * ~f[j2][m] * ~f[k2][n + 1]
						* f[j2][m] * f[k2][n] * f[i][n + 1];
				}
			}
		}
		if (n != m + 1) {
			if (m < nmax) {
				E5j1j2df -= 0.5 * (J[j1] * J[i] * cos2th / eps(U, j1, i, m, n))
					* g(m, n) * g(n - 1, m + 1) * ~f[j1][m + 1] * ~f[j2][m]
					* f[j1][m] * f[j2][m + 1] * f[i][n];
				E5j1j2df -= 0.5 * (J[i] * J[j1] * cos2th / eps(U, j2, i, m, n))
					* g(m, n) * g(n - 1, m + 1) * ~f[j2][m + 1] * ~f[j1][m]
					* f[j2][m] * f[j1][m + 1] * f[i][n];
			}
			if (n > 0) {
				if (m < nmax) {
					E5j1k1df += 0.5
						* (J[k1] * J[j1] * cos2th / eps(U, j1, k1, m, n))
						* g(m, n) * g(n - 1, m + 1) * ~f[j1][m + 1]
						* ~f[k1][n - 1] * f[j1][m + 1] * f[k1][n] * f[i][n - 1];
					E5j2k2df += 0.5
						* (J[j2] * J[i] * cos2th / eps(U, j2, k2, m, n))
						* g(m, n) * g(n - 1, m + 1) * ~f[j2][m + 1]
						* ~f[k2][n - 1] * f[j2][m + 1] * f[k2][n] * f[i][n - 1];
					E5j1k1df -= 0.5
						* (J[k1] * J[j1] * cos2th / eps(U, j1, k1, m, n))
						* g(m, n) * g(n - 1, m + 1) * ~f[j1][m] * ~f[k1][n - 1]
						* f[j1][m] * f[k1][n] * f[i][n - 1];
					E5j2k2df -= 0.5
						* (J[j2] * J[i] * cos2th / eps(U, j2, k2, m, n))
						* g(m, n) * g(n - 1, m + 1) * ~f[j2][m] * ~f[k2][n - 1]
						* f[j2][m] * f[k2][n] * f[i][n - 1];
				}
			}
		}
		if (n != m - 1) {
			if (n < nmax) {
				if (m > 0) {
					E5j1k1df += 0.5
						* (J[k1] * J[j1] * cos2th / eps(U, k1, j1, n, m))
						* g(n, m) * g(m - 1, n + 1) * ~f[k1][n + 1]
						* ~f[j1][m - 1] * f[k1][n] * f[j1][m - 1] * f[i][n + 1];
					E5j2k2df += 0.5
						* (J[j2] * J[i] * cos2th / eps(U, k2, j2, n, m))
						* g(n, m) * g(m - 1, n + 1) * ~f[k2][n + 1]
						* ~f[j2][m - 1] * f[k2][n] * f[j2][m - 1] * f[i][n + 1];
					E5j1j2df -= 0.5
						* (J[j1] * J[i] * cos2th / eps(U, i, j1, n, m))
						* g(n, m) * g(m - 1, n + 1) * ~f[j1][m - 1] * ~f[j2][m]
						* f[j1][m] * f[j2][m - 1] * f[i][n];
					E5j1j2df -= 0.5
						* (J[i] * J[j1] * cos2th / eps(U, i, j2, n, m))
						* g(n, m) * g(m - 1, n + 1) * ~f[j2][m - 1] * ~f[j1][m]
						* f[j2][m] * f[j1][m - 1] * f[i][n];
				}
				E5j1k1df -= 0.5
					* (J[k1] * J[j1] * cos2th / eps(U, k1, j1, n, m)) * g(n, m)
					* g(m - 1, n + 1) * ~f[k1][n + 1] * ~f[j1][m] * f[k1][n]
					* f[j1][m] * f[i][n + 1];
				E5j2k2df -= 0.5 * (J[j2] * J[i] * cos2th / eps(U, k2, j2, n, m))
					* g(n, m) * g(m - 1, n + 1) * ~f[k2][n + 1] * ~f[j2][m]
					* f[k2][n] * f[j2][m] * f[i][n + 1];
			}
		}
	}

	doublecomplex Edf = doublecomplex::zero();

	Edf += (E0df * norm2[i] - E0[i] * f[i][n]) / (norm2[i] * norm2[i]);

	Edf += (E1j1df * norm2[i] * norm2[j1]
		- 1 * (E1j1[i] + E1j2[j1]) * f[i][n] * norm2[j1])
		/ (norm2[i] * norm2[i] * norm2[j1] * norm2[j1]);
	Edf += (E1j2df * norm2[i] * norm2[j2]
		- 1 * (E1j2[i] + E1j1[j2]) * f[i][n] * norm2[j2])
		/ (norm2[i] * norm2[i] * norm2[j2] * norm2[j2]);

	Edf += (E2j1df * norm2[i] * norm2[j1]
		- (E2j1[i] + E2j2[j1]) * f[i][n] * norm2[j1])
		/ (norm2[i] * norm2[i] * norm2[j1] * norm2[j1]);
	Edf += (E2j2df * norm2[i] * norm2[j2]
		- (E2j2[i] + E2j1[j2]) * f[i][n] * norm2[j2])
		/ (norm2[i] * norm2[i] * norm2[j2] * norm2[j2]);

	Edf += (E3j1df * norm2[i] * norm2[j1]
		- (E3j1[i] + E3j2[j1]) * f[i][n] * norm2[j1])
		/ (norm2[i] * norm2[i] * norm2[j1] * norm2[j1]);
	Edf += (E3j2df * norm2[i] * norm2[j2]
		- (E3j2[i] + E3j1[j2]) * f[i][n] * norm2[j2])
		/ (norm2[i] * norm2[i] * norm2[j2] * norm2[j2]);

	Edf += (E4j1j2df * norm2[i] * norm2[j1] * norm2[j2]
		- (E4j1j2[i] + E4j2k2[j1] + E4j1k1[j2]) * f[i][n] * norm2[j1]
			* norm2[j2])
		/ (norm2[i] * norm2[i] * norm2[j1] * norm2[j1] * norm2[j2] * norm2[j2]);
	Edf += (E4j1k1df * norm2[i] * norm2[j1] * norm2[k1]
		- (E4j1k1[i] + E4j1j2[j1] + E4j2k2[k1]) * f[i][n] * norm2[j1]
			* norm2[k1])
		/ (norm2[i] * norm2[i] * norm2[j1] * norm2[j1] * norm2[k1] * norm2[k1]);
	Edf += (E4j2k2df * norm2[i] * norm2[j2] * norm2[k2]
		- (E4j2k2[i] + E4j1j2[j2] + E4j1k1[k2]) * f[i][n] * norm2[j2]
			* norm2[k2])
		/ (norm2[i] * norm2[i] * norm2[j2] * norm2[j2] * norm2[k2] * norm2[k2]);

	Edf += (E5j1j2df * norm2[i] * norm2[j1] * norm2[j2]
		- (E5j1j2[i] + E5j2k2[j1] + E5j1k1[j2]) * f[i][n] * norm2[j1]
			* norm2[j2])
		/ (norm2[i] * norm2[i] * norm2[j1] * norm2[j1] * norm2[j2] * norm2[j2]);
	Edf += (E5j1k1df * norm2[i] * norm2[j1] * norm2[k1]
		- (E5j1k1[i] + E5j1j2[j1] + E5j2k2[k1]) * f[i][n] * norm2[j1]
			* norm2[k1])
		/ (norm2[i] * norm2[i] * norm2[j1] * norm2[j1] * norm2[k1] * norm2[k1]);
	Edf += (E5j2k2df * norm2[i] * norm2[j2] * norm2[k2]
		- (E5j2k2[i] + E5j1j2[j2] + E5j1k1[k2]) * f[i][n] * norm2[j2]
			* norm2[k2])
		/ (norm2[i] * norm2[i] * norm2[j2] * norm2[j2] * norm2[k2] * norm2[k2]);

	int k = i * dim + n;
	g_dev[2 * k] = 2 * Edf.real();
	g_dev[2 * k + 1] = 2 * Edf.imag();
}

__global__ void norm2Ker(real* x, real* norm2s) {
	int i = threadIdx.x;

	if (i >= L) {
		return;
	}

	__shared__ doublecomplex fi[L * dim];
	__shared__ doublecomplex* f[L];
	if (i == 0) {
		for (int j = 0; j < L; j++) {
			int k = j * dim;
			f[j] = &fi[k];
			for (int m = 0; m <= nmax; m++) {
				f[j][m] = make_doublecomplex(x[2 * (k + m)],
					x[2 * (k + m) + 1]);
			}
		}
	}
	__syncthreads();

	norm2s[i] = 0;
	for (int n = 0; n <= nmax; n++) {
		norm2s[i] += norm2(f[i][n]);
	}
}

__global__ void copyfker(doublecomplex* E, real* f) {
	*f = E->real();
}

void zero(Estruct* Es_dev) {
	Estruct Es;
	memCopy(&Es, Es_dev, sizeof(Estruct), cudaMemcpyDeviceToHost);

	cudaMemset(Es.E0, 0, L * sizeof(doublecomplex));
	cudaMemset(Es.E1j1, 0, L * sizeof(doublecomplex));
	cudaMemset(Es.E1j2, 0, L * sizeof(doublecomplex));
	cudaMemset(Es.E2j1, 0, L * sizeof(doublecomplex));
	cudaMemset(Es.E2j2, 0, L * sizeof(doublecomplex));
	cudaMemset(Es.E3j1, 0, L * sizeof(doublecomplex));
	cudaMemset(Es.E3j2, 0, L * sizeof(doublecomplex));
	cudaMemset(Es.E4j1j2, 0, L * sizeof(doublecomplex));
	cudaMemset(Es.E4j1k1, 0, L * sizeof(doublecomplex));
	cudaMemset(Es.E4j2k2, 0, L * sizeof(doublecomplex));
	cudaMemset(Es.E5j1j2, 0, L * sizeof(doublecomplex));
	cudaMemset(Es.E5j1k1, 0, L * sizeof(doublecomplex));
	cudaMemset(Es.E5j2k2, 0, L * sizeof(doublecomplex));
}

void allocE(Estruct** Es_dev) {
	Estruct Es;
	CudaSafeMemAllocCall(memAlloc<doublecomplex>(&Es.E0, L));
	CudaSafeMemAllocCall(memAlloc<doublecomplex>(&Es.E1j1, L));
	CudaSafeMemAllocCall(memAlloc<doublecomplex>(&Es.E1j2, L));
	CudaSafeMemAllocCall(memAlloc<doublecomplex>(&Es.E2j1, L));
	CudaSafeMemAllocCall(memAlloc<doublecomplex>(&Es.E2j2, L));
	CudaSafeMemAllocCall(memAlloc<doublecomplex>(&Es.E3j1, L));
	CudaSafeMemAllocCall(memAlloc<doublecomplex>(&Es.E3j2, L));
	CudaSafeMemAllocCall(memAlloc<doublecomplex>(&Es.E4j1j2, L));
	CudaSafeMemAllocCall(memAlloc<doublecomplex>(&Es.E4j1k1, L));
	CudaSafeMemAllocCall(memAlloc<doublecomplex>(&Es.E4j2k2, L));
	CudaSafeMemAllocCall(memAlloc<doublecomplex>(&Es.E5j1j2, L));
	CudaSafeMemAllocCall(memAlloc<doublecomplex>(&Es.E5j1k1, L));
	CudaSafeMemAllocCall(memAlloc<doublecomplex>(&Es.E5j2k2, L));

	CudaSafeMemAllocCall(memAlloc<Estruct>(Es_dev, 1));
	memCopy(*Es_dev, &Es, sizeof(Estruct), cudaMemcpyHostToDevice);
}

void freeE(Estruct* Es_dev) {
	Estruct Es;
	memCopy(&Es, Es_dev, sizeof(Estruct), cudaMemcpyDeviceToHost);

	memFree(Es.E0);
	memFree(Es.E1j1);
	memFree(Es.E1j2);
	memFree(Es.E2j1);
	memFree(Es.E2j2);
	memFree(Es.E3j1);
	memFree(Es.E3j2);
	memFree(Es.E4j1j2);
	memFree(Es.E4j1k1);
	memFree(Es.E4j2k2);
	memFree(Es.E5j1j2);
	memFree(Es.E5j1k1);
	memFree(Es.E5j2k2);

	memFree(Es_dev);
}

void allocWorkspace(Workspace& work) {
	memAlloc<real>(&work.norm2, L);
}

void freeWorkspace(Workspace& work) {
	memFree(work.norm2);
}

void energy(real* x, real* f_dev, real* g_dev, Parameters& parms, real theta, Estruct* Es,
	Workspace& work) {

	doublecomplex* E_dev;
	CudaSafeMemAllocCall(memAlloc<doublecomplex>(&E_dev, 1));
	cudaMemset(E_dev, 0, sizeof(doublecomplex));
	CudaCheckError();

	zero(Es);
	CudaCheckError();

	norm2Ker<<<1, L>>>(x, work.norm2);
	CudaCheckError();
	energyfctKer<<<dim, L>>>(x, work.norm2, parms, theta, E_dev, Es);
	CudaCheckError();
	copyfker<<<1, 1>>>(E_dev, f_dev);
	CudaCheckError();
	energygctKer<<<dim, L>>>(x, work.norm2, parms, theta, Es, g_dev);
	CudaCheckError();

	doublecomplex E;
	memCopy(&E, E_dev, sizeof(doublecomplex), cudaMemcpyDeviceToHost);
//	printf("E = %f, %f\n", E.real(), E.imag());
	real* g_host = new real[2 * L * dim];
	memCopy(g_host, g_dev, 2 * L * dim * sizeof(real), cudaMemcpyDeviceToHost);
//	printf("g: ");
//	for (int i = 0; i < L; i++) {
//		for (int n = 0; n <= nmax; n++) {
//			printf("%f %f, ", g_host[2 * (i * dim + n)],
//				g_host[2 * (i * dim + n) + 1]);
//		}
//	}
//	printf("\n");

	memFree(E_dev);
}

