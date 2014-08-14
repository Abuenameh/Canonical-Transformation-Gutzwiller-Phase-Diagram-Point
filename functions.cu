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

__global__ void initProbKer(real* x, int* nbd, real* l, real* u) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= 2 * L * dim) {
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
	x[i] = (1 / sqrt(2.0 * dim)) * (k + n) / scale;
	nbd[i] = 0;
}

__global__ void initProbKer(real* x, int* nbd, real* l, real* u, int j,
	real dx) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= 2 * L * dim) {
		return;
	}

	int k = i / (2 * dim);
	int n = i % (2 * dim);
	x[i] = (1 / sqrt(2.0 * dim)) * (k + n);
	nbd[i] = 0;
	if (i == j)
		x[j] += dx;
}

extern void initProb(real* x, int* nbd, real* l, real* u) {
	initProbKer<<<lbfgsbcuda::iDivUp(L * dim, 64), 64>>>(x, nbd, l, u);
}

extern void initProb(real* x, int* nbd, real* l, real* u, int i, real dx) {
	initProbKer<<<lbfgsbcuda::iDivUp(L * dim, 64), 64>>>(x, nbd, l, u, i, dx);
}

__global__ void energyKer(real* x, real* f_dev, real* g_dev, real* U, real* J,
	real mu, real* norm2s) {

	if (threadIdx.x > 0) {
		return;
	}

	int i = blockIdx.x;
	if (i >= L) {
		return;
	}

	//	__shared__ doublecomplex f[L*dim];
	__shared__ doublecomplex fi[5 * dim];
	__shared__ doublecomplex* f[5];

	int k = i * dim;

	int k1 = mod(i - 2);
	int j1 = mod(i - 1);
	int j2 = mod(i + 1);
	int k2 = mod(i + 2);

	f[0] = &fi[k1 * dim];
	f[1] = &fi[j1 * dim];
	f[2] = &fi[i * dim];
	f[3] = &fi[j2 * dim];
	f[4] = &fi[k2 * dim];

	for (int n = 0; n <= nmax; n++) {
		f[0][n] = make_doublecomplex(x[2 * (k1 * dim + n)],
			x[2 * (k1 * dim + n) + 1]);
		f[1][n] = make_doublecomplex(x[2 * (j1 * dim + n)],
			x[2 * (j1 * dim + n) + 1]);
		f[2][n] = make_doublecomplex(x[2 * (i * dim + n)],
			x[2 * (i * dim + n) + 1]);
		f[3][n] = make_doublecomplex(x[2 * (j2 * dim + n)],
			x[2 * (j2 * dim + n) + 1]);
		f[4][n] = make_doublecomplex(x[2 * (k2 * dim + n)],
			x[2 * (k2 * dim + n) + 1]);
	}

	__shared__ doublecomplex norm2[5];
	norm2[0] = norm2s[k1];
	norm2[1] = norm2s[j1];
	norm2[2] = norm2s[i];
	norm2[3] = norm2s[j2];
	norm2[4] = norm2s[k2];
//	for (int j = 0; j < 5; j++) {
//		norm2[j] = doublecomplex::zero();
//		for(int n = 0; n <= nmax; n++) {
//			norm2[j] += f[j][n].abs() * f[j][n].abs();
//		}
//	}

	k1 = 0;
	j1 = 1;
	i = 2;
	j2 = 3;
	k2 = 4;

	doublecomplex E = doublecomplex::zero();
//	doublecomplex grad = doublecomplex::zero();
	__shared__ doublecomplex grad[dim];

	doublecomplex E0 = doublecomplex::zero();
	doublecomplex E0df = doublecomplex::zero();
	doublecomplex E1j1 = doublecomplex::zero();
//	doublecomplex E1j1df = doublecomplex::zero();
	doublecomplex E1j2 = doublecomplex::zero();
//	doublecomplex E1j2df = doublecomplex::zero();
	doublecomplex E1j1df[dim];
	doublecomplex E1j2df[dim];

	for (int n = 0; n <= nmax; n++) {
//		doublecomplex E1 = doublecomplex::zero();
		E0 = (0.5 * U[i] * n * (n - 1) - mu * n) * ~f[i][n] * f[i][n];
		E0df = (U[i] * n * (n - 1) - 2 * mu * n) * f[i][n];
//		E += E0 / norm2[i];
//		grad += (E0df * norm2[i] - E0 * 2 * f[i][n]) / (norm2[i] * norm2[i]);

//		E1j1 = doublecomplex::zero();
		E1j1df[n] = doublecomplex::zero();
//		E1j2 = doublecomplex::zero();
		E1j2df[n] = doublecomplex::zero();
		if (n < nmax) {
			E1j1 += -J[j1] * g(n, n + 1) * ~f[i][n + 1] * ~f[j1][n] * f[i][n]
				* f[j1][n + 1];
			E1j2 += -J[i] * g(n, n + 1) * ~f[i][n + 1] * ~f[j2][n] * f[i][n]
				* f[j2][n + 1];
			E1j1df[n] += -2 * J[j1] * g(n, n + 1) * ~f[j1][n + 1] * f[j1][n]
				* f[i][n + 1];
			E1j2df[n] += -2 * J[i] * g(n, n + 1) * ~f[j2][n + 1] * f[j2][n]
				* f[i][n + 1];
		}
		if (n > 0) {
			E1j1df[n] += -2 * J[j1] * g(n - 1, n) * ~f[j1][n - 1] * f[j1][n]
				* f[i][n - 1];
			E1j2df[n] += -2 * J[i] * g(n - 1, n) * ~f[j2][n - 1] * f[j2][n]
				* f[i][n - 1];
		}
//		printf("E1j1df1[%d,%d]: %f, %f\n", blockIdx.x, n, E1j1df.real(),
//			E1j1df.imag());
//		printf("E1j2df1[%d,%d]: %f, %f\n", blockIdx.x, n, E1j2df.real(),
//			E1j2df.imag());
		grad[n] = 0;

//		E += E0 / norm2[i];
//		grad[n] += (E0df * norm2[i] - E0 * 2 * f[i][n]) / (norm2[i] * norm2[i]);
//		E += E1j1 / (norm2[i] * norm2[j1]);
//		grad[n] += (E1j1df * norm2[i] * norm2[j1] - E1j1 * 2 * f[i][n])
//			/ (norm2[i] * norm2[i] * norm2[j1] * norm2[j1]);
//		E += E1j2 / (norm2[i] * norm2[j2]);
//		grad[n] += (E1j2df * norm2[i] * norm2[j2] - E1j2 * 2 * f[i][n])
//			/ (norm2[i] * norm2[i] * norm2[j2] * norm2[j2]);

//		g_dev[2 * (k + n)] = grad[n].real();
//		g_dev[2 * (k + n) + 1] = grad[n].imag();
	}

	E += E1j1 / (norm2[i] * norm2[j1]);
	E += E1j2 / (norm2[i] * norm2[j2]);

	//	printf("E1j1 = %f, %f\nE1j2 = %f, %f\n", E1j1.real(), E1j1.imag(), E1j2.real(), E1j2.imag());
	for (int n = 0; n <= nmax; n++) {
//		printf("E1j1df[%d] = %f, %f\n", n, E1j1df[n].real(), E1j1df[n].imag());
		grad[n] = 0;
		grad[n] += (E1j1df[n] * norm2[i] * norm2[j1] - 2 * E1j1 * 2 * f[i][n])
			/ (norm2[i] * norm2[i] * norm2[j1] * norm2[j1]);
		grad[n] += (E1j2df[n] * norm2[i] * norm2[j2] - 2 * E1j2 * 2 * f[i][n])
			/ (norm2[i] * norm2[i] * norm2[j2] * norm2[j2]);
		double qwe = ((E1j1df[n] * norm2[i] * norm2[j1] - E1j1 * 2 * f[i][n])
			/ (norm2[i] * norm2[i] * norm2[j1] * norm2[j1])).real();
		double asd = ((E1j2df[n] * norm2[i] * norm2[j2] - E1j2 * 2 * f[i][n])
			/ (norm2[i] * norm2[i] * norm2[j2] * norm2[j2])).real();
//		printf("qwe[%d] = %f\nasd[%d] = %f\n", n, qwe, n, asd);

		g_dev[2 * (k + n)] = grad[n].real();
		g_dev[2 * (k + n) + 1] = grad[n].imag();
	}
//	printf("E[%d] = %f\n", blockIdx.x, E.real());

	atomicAdd(f_dev, E.real());
}

__global__ void energyfKer(real* x, doublecomplex* E0_dev,
	doublecomplex* E1j1_dev, doublecomplex* E1j2_dev, real* U, real* J, real mu,
	real* norm2s) {

	if (threadIdx.x > 0) {
		return;
	}

	int idx = blockIdx.x;
	if (idx >= L) {
		return;
	}

	__shared__ doublecomplex fi[5 * dim];
	__shared__ doublecomplex* f[5];

	int k = idx * dim;

	int k1 = mod(idx - 2);
	int j1 = mod(idx - 1);
	int i = idx;
	int j2 = mod(idx + 1);
	int k2 = mod(idx + 2);

	f[0] = &fi[k1 * dim];
	f[1] = &fi[j1 * dim];
	f[2] = &fi[i * dim];
	f[3] = &fi[j2 * dim];
	f[4] = &fi[k2 * dim];

	for (int n = 0; n <= nmax; n++) {
		f[0][n] = make_doublecomplex(x[2 * (k1 * dim + n)],
			x[2 * (k1 * dim + n) + 1]);
		f[1][n] = make_doublecomplex(x[2 * (j1 * dim + n)],
			x[2 * (j1 * dim + n) + 1]);
		f[2][n] = make_doublecomplex(x[2 * (i * dim + n)],
			x[2 * (i * dim + n) + 1]);
		f[3][n] = make_doublecomplex(x[2 * (j2 * dim + n)],
			x[2 * (j2 * dim + n) + 1]);
		f[4][n] = make_doublecomplex(x[2 * (k2 * dim + n)],
			x[2 * (k2 * dim + n) + 1]);
	}

	__shared__ doublecomplex norm2[5];
	norm2[0] = norm2s[k1];
	norm2[1] = norm2s[j1];
	norm2[2] = norm2s[i];
	norm2[3] = norm2s[j2];
	norm2[4] = norm2s[k2];
//	for (int j = 0; j < 5; j++) {
//		norm2[j] = doublecomplex::zero();
//		for(int n = 0; n <= nmax; n++) {
//			norm2[j] += f[j][n].abs() * f[j][n].abs();
//		}
//	}

	k1 = 0;
	j1 = 1;
	i = 2;
	j2 = 3;
	k2 = 4;

	doublecomplex E = doublecomplex::zero();

	doublecomplex E0 = doublecomplex::zero();
	doublecomplex E1j1 = doublecomplex::zero();
	doublecomplex E1j2 = doublecomplex::zero();

	for (int n = 0; n <= nmax; n++) {
		E0 = (0.5 * U[i] * n * (n - 1) - mu * n) * ~f[i][n] * f[i][n];

		if (n < nmax) {
			E1j1 += -J[j1] * g(n, n + 1) * ~f[i][n + 1] * ~f[j1][n] * f[i][n]
				* f[j1][n + 1];
			E1j2 += -J[i] * g(n, n + 1) * ~f[i][n + 1] * ~f[j2][n] * f[i][n]
				* f[j2][n + 1];
		}
	}

	E += E1j1 / (norm2[i] * norm2[j1]);
	E += E1j2 / (norm2[i] * norm2[j2]);

	E0_dev[idx] = E0;
	E1j1_dev[idx] = E1j1;
	E1j2_dev[idx] = E1j2;

//	atomicAdd(f_dev, E.real());
}

__global__ void energygKer(real* x, real* f_dev, real* g_dev, doublecomplex* E0,
	doublecomplex* E1j1, doublecomplex* E1j2, real* U, real* J, real mu,
	real* norm2s) {

	if (threadIdx.x > 0) {
		return;
	}

	int idx = blockIdx.x;
	if (idx >= L) {
		return;
	}

	//	__shared__ doublecomplex f[L*dim];
	__shared__ doublecomplex fi[5 * dim];
	__shared__ doublecomplex* f[5];

	int k = idx * dim;

	int k1 = mod(idx - 2);
	int j1 = mod(idx - 1);
	int i = idx;
	int j2 = mod(idx + 1);
	int k2 = mod(idx + 2);

	int iorig = i;
	int k1orig = k1;
	int j1orig = j1;
	int j2orig = j2;
	int k2orig = k2;

	f[0] = &fi[k1 * dim];
	f[1] = &fi[j1 * dim];
	f[2] = &fi[i * dim];
	f[3] = &fi[j2 * dim];
	f[4] = &fi[k2 * dim];

	for (int n = 0; n <= nmax; n++) {
		f[0][n] = make_doublecomplex(x[2 * (k1 * dim + n)],
			x[2 * (k1 * dim + n) + 1]);
		f[1][n] = make_doublecomplex(x[2 * (j1 * dim + n)],
			x[2 * (j1 * dim + n) + 1]);
		f[2][n] = make_doublecomplex(x[2 * (i * dim + n)],
			x[2 * (i * dim + n) + 1]);
		f[3][n] = make_doublecomplex(x[2 * (j2 * dim + n)],
			x[2 * (j2 * dim + n) + 1]);
		f[4][n] = make_doublecomplex(x[2 * (k2 * dim + n)],
			x[2 * (k2 * dim + n) + 1]);
	}

	__shared__ doublecomplex norm2[5];
	norm2[0] = norm2s[k1];
	norm2[1] = norm2s[j1];
	norm2[2] = norm2s[i];
	norm2[3] = norm2s[j2];
	norm2[4] = norm2s[k2];
//	for (int j = 0; j < 5; j++) {
//		norm2[j] = doublecomplex::zero();
//		for(int n = 0; n <= nmax; n++) {
//			norm2[j] += f[j][n].abs() * f[j][n].abs();
//		}
//	}

	k1 = 0;
	j1 = 1;
	i = 2;
	j2 = 3;
	k2 = 4;

	doublecomplex E = doublecomplex::zero();
//	doublecomplex grad = doublecomplex::zero();
	__shared__ doublecomplex grad[dim];

	doublecomplex E0df = doublecomplex::zero();
//	doublecomplex E1j1df = doublecomplex::zero();
//	doublecomplex E1j2df = doublecomplex::zero();
	doublecomplex E1j1df[dim];
	doublecomplex E1j2df[dim];

	for (int n = 0; n <= nmax; n++) {
		E0df = (U[i] * n * (n - 1) - 2 * mu * n) * f[i][n];
//		E += E0 / norm2[i];
//		grad += (E0df * norm2[i] - E0 * 2 * f[i][n]) / (norm2[i] * norm2[i]);

		E1j1df[n] = doublecomplex::zero();
		E1j2df[n] = doublecomplex::zero();
		if (n < nmax) {
			E1j1df[n] += -2 * J[j1] * g(n, n + 1) * ~f[j1][n + 1] * f[j1][n]
				* f[i][n + 1];
			E1j2df[n] += -2 * J[i] * g(n, n + 1) * ~f[j2][n + 1] * f[j2][n]
				* f[i][n + 1];
		}
		if (n > 0) {
			E1j1df[n] += -2 * J[j1] * g(n - 1, n) * ~f[j1][n - 1] * f[j1][n]
				* f[i][n - 1];
			E1j2df[n] += -2 * J[i] * g(n - 1, n) * ~f[j2][n - 1] * f[j2][n]
				* f[i][n - 1];
		}
	}

	E += E1j1[i] / (norm2[i] * norm2[j1]);
	E += E1j2[i] / (norm2[i] * norm2[j2]);

	//	printf("E1j1 = %f, %f\nE1j2 = %f, %f\n", E1j1.real(), E1j1.imag(), E1j2.real(), E1j2.imag());
	for (int n = 0; n <= nmax; n++) {
//		printf("E1j1df[%d] = %f, %f\n", n, E1j1df[n].real(), E1j1df[n].imag());
		grad[n] = 0;
		grad[n] += (E1j1df[n] * norm2[i] * norm2[j1]
			- (E1j1[iorig] + E1j2[j1orig]) * 2 * f[i][n])
			/ (norm2[i] * norm2[i] * norm2[j1] * norm2[j1]);
		grad[n] += (E1j2df[n] * norm2[i] * norm2[j2]
			- (E1j2[iorig] + E1j1[j2orig]) * 2 * f[i][n])
			/ (norm2[i] * norm2[i] * norm2[j2] * norm2[j2]);
//		printf("E1j1df[%d] = %f\nE1j2[%d] = %f\n", n, E1j1df[n].real(), i, E1j2[j1orig].real());
//		printf("grad[%d] = %f\n", n, grad[n]);

		g_dev[2 * (k + n)] = grad[n].real();
		g_dev[2 * (k + n) + 1] = grad[n].imag();
	}
//	printf("E[%d] = %f\n", blockIdx.x, E.real());

	atomicAdd(f_dev, E.real());
}

struct Estruct {
	doublecomplex* E0;
	doublecomplex* E1j1;
	doublecomplex* E1j2;
	doublecomplex* E2j1;
	doublecomplex* E2j2;
	doublecomplex* E3j1;
	doublecomplex* E3j2;
	doublecomplex* E4j1j2;
	doublecomplex* E4j1k1;
	doublecomplex* E4j2k2;
	doublecomplex* E5j1j2;
	doublecomplex* E5j1k1;
	doublecomplex* E5j2k2;
};

__global__ void testKer(real* x, real* norm2, real* U_dev, real* J_dev, real mu,
	doublecomplex* E, Estruct Es) {
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
	if (i == 0) {
		for (int j = 0; j < L; j++) {
			int k = j * dim;
			f[j] = &fi[k];
			for (int m = 0; m <= nmax; m++) {
				f[j][m] = make_doublecomplex(x[2 * (k + m)],
					x[2 * (k + m) + 1]);
			}
			U[j] = U_dev[j];
			J[j] = J_dev[j];
		}
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
		E1j1 += -J[j1] * g(n, n + 1) * ~f[i][n + 1] * ~f[j1][n] * f[i][n]
			* f[j1][n + 1];
		E1j2 += -J[i] * g(n, n + 1) * ~f[i][n + 1] * ~f[j2][n] * f[i][n]
			* f[j2][n + 1];

		if (n > 0) {
			E2j1 += 0.5 * J[j1] * J[j1] * g(n, n) * g(n - 1, n + 1)
				* ~f[i][n + 1] * ~f[j1][n - 1] * f[i][n - 1] * f[j1][n + 1]
				* (1 / eps(U, i, j1, n, n) - 1 / eps(U, i, j1, n - 1, n + 1));
			E2j2 += 0.5 * J[i] * J[i] * g(n, n) * g(n - 1, n + 1) * ~f[i][n + 1]
				* ~f[j2][n - 1] * f[i][n - 1] * f[j2][n + 1]
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
				E5j1j2 += 0.5 * (J[j1] * J[i] / eps(U, i, j1, n, m)) * g(n, m)
					* g(m - 1, n + 1) * ~f[i][n + 1] * ~f[j1][m - 1] * ~f[j2][m]
					* f[i][n + 1] * f[j1][m] * f[j2][m - 1];
				E5j1j2 += 0.5 * (J[i] * J[j1] / eps(U, i, j2, n, m)) * g(n, m)
					* g(m - 1, n + 1) * ~f[i][n + 1] * ~f[j2][m - 1] * ~f[j1][m]
					* f[i][n + 1] * f[j2][m] * f[j1][m - 1];
				E5j1k1 += 0.5 * (J[j1] * J[k1] / eps(U, i, j1, n, m)) * g(n, m)
					* g(m - 1, n + 1) * ~f[i][n + 1] * ~f[j1][m - 1] * ~f[k1][n]
					* f[i][n] * f[j1][m - 1] * f[k1][n + 1];
				E5j2k2 += 0.5 * (J[i] * J[j2] / eps(U, i, j2, n, m)) * g(n, m)
					* g(m - 1, n + 1) * ~f[i][n + 1] * ~f[j2][m - 1] * ~f[k2][n]
					* f[i][n] * f[j2][m - 1] * f[k2][n + 1];
				E5j1j2 -= 0.5 * (J[j1] * J[i] / eps(U, i, j1, n, m)) * g(n, m)
					* g(m - 1, n + 1) * ~f[i][n] * ~f[j1][m - 1] * ~f[j2][m]
					* f[i][n] * f[j1][m] * f[j2][m - 1];
				E5j1j2 -= 0.5 * (J[i] * J[j1] / eps(U, i, j2, n, m)) * g(n, m)
					* g(m - 1, n + 1) * ~f[i][n] * ~f[j2][m - 1] * ~f[j1][m]
					* f[i][n] * f[j2][m] * f[j1][m - 1];
				E5j1k1 -= 0.5 * (J[j1] * J[k1] / eps(U, i, j1, n, m)) * g(n, m)
					* g(m - 1, n + 1) * ~f[i][n + 1] * ~f[j1][m] * ~f[k1][n]
					* f[i][n] * f[j1][m] * f[k1][n + 1];
				E5j2k2 -= 0.5 * (J[i] * J[j2] / eps(U, i, j2, n, m)) * g(n, m)
					* g(m - 1, n + 1) * ~f[i][n + 1] * ~f[j2][m] * ~f[k2][n]
					* f[i][n] * f[j2][m] * f[k2][n + 1];
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

	atomicAdd(&Es.E0[i], E0);
	atomicAdd(&Es.E1j1[i], E1j1);
	atomicAdd(&Es.E1j2[i], E1j2);
	atomicAdd(&Es.E2j1[i], E2j1);
	atomicAdd(&Es.E2j2[i], E2j2);
	atomicAdd(&Es.E3j1[i], E3j1);
	atomicAdd(&Es.E3j2[i], E3j2);
	atomicAdd(&Es.E4j1j2[i], E4j1j2);
	atomicAdd(&Es.E4j1k1[i], E4j1k1);
	atomicAdd(&Es.E4j2k2[i], E4j2k2);
	atomicAdd(&Es.E5j1j2[i], E5j1j2);
	atomicAdd(&Es.E5j1k1[i], E5j1k1);
	atomicAdd(&Es.E5j2k2[i], E5j2k2);

	//	if(blockIdx.x>2) {
//		return;
//	}
//	if(threadIdx.x>10) {
//		return;
//	}
//__shared__ doublecomplex test[L*dim];
//printf("L=%d, dim=%d, test[%d,%d]=%f\n",L,dim,blockIdx.x,threadIdx.x,test[0].real());
}

struct Edfstruct {
	doublecomplex** E0df;
	doublecomplex** E1j1df;
	doublecomplex** E1j2df;
	doublecomplex** E2j1df;
	doublecomplex** E2j2df;
	doublecomplex** E3j1df;
	doublecomplex** E3j2df;
};

__global__ void testKer(real* x, real* norm2, real* g_dev, real* U_dev,
	real* J_dev, real mu, doublecomplex* E, Estruct Es, Edfstruct Edfs) {
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
			U[j] = U_dev[j];
			J[j] = J_dev[j];
			E0[j] = Es.E0[j];
			E1j1[j] = Es.E1j1[j];
			E1j2[j] = Es.E1j2[j];
			E2j1[j] = Es.E2j1[j];
			E2j2[j] = Es.E2j2[j];
			E3j1[j] = Es.E3j1[j];
			E3j2[j] = Es.E3j2[j];
			E4j1j2[j] = Es.E4j1j2[j];
			E4j1k1[j] = Es.E4j1k1[j];
			E4j2k2[j] = Es.E4j2k2[j];
			E5j1j2[j] = Es.E5j1j2[j];
			E5j1k1[j] = Es.E5j1k1[j];
			E5j2k2[j] = Es.E5j2k2[j];
		}
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
		E1j1df += -J[j1] * g(n, n + 1) * ~f[j1][n + 1] * f[j1][n] * f[i][n + 1];
		E1j2df += -J[i] * g(n, n + 1) * ~f[j2][n + 1] * f[j2][n] * f[i][n + 1];
	}
	if (n > 0) {
		E1j1df += -J[j1] * g(n - 1, n) * ~f[j1][n - 1] * f[j1][n] * f[i][n - 1];
		E1j2df += -J[i] * g(n - 1, n) * ~f[j2][n - 1] * f[j2][n] * f[i][n - 1];
	}

	if (n > 1) {
		E2j1df += 0.5 * J[j1] * J[j1] * g(n - 1, n - 1) * g(n - 2, n)
			* ~f[j1][n - 2] * f[j1][n] * f[i][n - 2]
			* (1 / eps(U, i, j1, n - 1, n - 1) - 1 / eps(U, i, j1, n - 2, n));
		E2j2df += 0.5 * J[i] * J[i] * g(n - 1, n - 1) * g(n - 2, n)
			* ~f[j2][n - 2] * f[j2][n] * f[i][n - 2]
			* (1 / eps(U, i, j2, n - 1, n - 1) - 1 / eps(U, i, j2, n - 2, n));
	}
	if (n < nmax - 1) {
		E2j1df += 0.5 * J[j1] * J[j1] * g(n + 1, n + 1) * g(n, n + 2)
			* ~f[j1][n + 2] * f[j1][n] * f[i][n + 2]
			* (1 / eps(U, j1, i, n + 1, n + 1) - 1 / eps(U, j1, i, n, n + 2));
		E2j2df += 0.5 * J[i] * J[i] * g(n + 1, n + 1) * g(n, n + 2)
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
					E5j1j2df += 0.5 * (J[j1] * J[i] / eps(U, i, j1, n - 1, m))
						* g(n - 1, m) * g(m - 1, n) * ~f[j1][m - 1] * ~f[j2][m]
						* f[j1][m] * f[j2][m - 1] * f[i][n];
					E5j1j2df += 0.5 * (J[i] * J[j1] / eps(U, i, j2, n - 1, m))
						* g(n - 1, m) * g(m - 1, n) * ~f[j2][m - 1] * ~f[j1][m]
						* f[j2][m] * f[j1][m - 1] * f[i][n];
					E5j1k1df += 0.5 * (J[j1] * J[k1] / eps(U, i, j1, n - 1, m))
						* g(n - 1, m) * g(m - 1, n) * ~f[j1][m - 1]
						* ~f[k1][n - 1] * f[j1][m - 1] * f[k1][n] * f[i][n - 1];
					E5j2k2df += 0.5 * (J[i] * J[j2] / eps(U, i, j2, n - 1, m))
						* g(n - 1, m) * g(m - 1, n) * ~f[j2][m - 1]
						* ~f[k2][n - 1] * f[j2][m - 1] * f[k2][n] * f[i][n - 1];
					E5j1k1df -= 0.5 * (J[j1] * J[k1] / eps(U, i, j1, n - 1, m))
						* g(n - 1, m) * g(m - 1, n) * ~f[j1][m] * ~f[k1][n - 1]
						* f[j1][m] * f[k1][n] * f[i][n - 1];
					E5j2k2df -= 0.5 * (J[i] * J[j2] / eps(U, i, j2, n - 1, m))
						* g(n - 1, m) * g(m - 1, n) * ~f[j2][m] * ~f[k2][n - 1]
						* f[j2][m] * f[k2][n] * f[i][n - 1];
				}
			}
			if (n < nmax) {
				if (m < nmax) {
					E5j1k1df += 0.5 * (J[j1] * J[k1] / eps(U, j1, i, m, n + 1))
						* g(m, n + 1) * g(n, m + 1) * ~f[j1][m + 1]
						* ~f[k1][n + 1] * f[j1][m + 1] * f[k1][n] * f[i][n + 1];
					E5j2k2df += 0.5 * (J[i] * J[j2] / eps(U, j2, i, m, n + 1))
						* g(m, n + 1) * g(n, m + 1) * ~f[j2][m + 1]
						* ~f[k2][n + 1] * f[j2][m + 1] * f[k2][n] * f[i][n + 1];
					E5j1j2df += 0.5 * (J[j1] * J[i] / eps(U, j1, i, m, n + 1))
						* g(m, n + 1) * g(n, m + 1) * ~f[j1][m + 1] * ~f[j2][m]
						* f[j1][m] * f[j2][m + 1] * f[i][n];
					E5j1j2df += 0.5 * (J[i] * J[j1] / eps(U, j2, i, m, n + 1))
						* g(m, n + 1) * g(n, m + 1) * ~f[j2][m + 1] * ~f[j1][m]
						* f[j2][m] * f[j1][m + 1] * f[i][n];
					E5j1k1df -= 0.5 * (J[j1] * J[k1] / eps(U, j1, i, m, n + 1))
						* g(m, n + 1) * g(n, m + 1) * ~f[j1][m] * ~f[k1][n + 1]
						* f[j1][m] * f[k1][n] * f[i][n + 1];
					E5j2k2df -= 0.5 * (J[i] * J[j2] / eps(U, j2, i, m, n + 1))
						* g(m, n + 1) * g(n, m + 1) * ~f[j2][m] * ~f[k2][n + 1]
						* f[j2][m] * f[k2][n] * f[i][n + 1];
				}
			}
		}
		if (n != m + 1) {
			if (m < nmax) {
				E5j1j2df -= 0.5 * (J[j1] * J[i] / eps(U, j1, i, m, n)) * g(m, n)
					* g(n - 1, m + 1) * ~f[j1][m + 1] * ~f[j2][m] * f[j1][m]
					* f[j2][m + 1] * f[i][n];
				E5j1j2df -= 0.5 * (J[i] * J[j1] / eps(U, j2, i, m, n)) * g(m, n)
					* g(n - 1, m + 1) * ~f[j2][m + 1] * ~f[j1][m] * f[j2][m]
					* f[j1][m + 1] * f[i][n];
			}
			if (n > 0) {
				if (m < nmax) {
					E5j1k1df += 0.5 * (J[k1] * J[j1] / eps(U, j1, k1, m, n))
						* g(m, n) * g(n - 1, m + 1) * ~f[j1][m + 1]
						* ~f[k1][n - 1] * f[j1][m + 1] * f[k1][n] * f[i][n - 1];
					E5j2k2df += 0.5 * (J[j2] * J[i] / eps(U, j2, k2, m, n))
						* g(m, n) * g(n - 1, m + 1) * ~f[j2][m + 1]
						* ~f[k2][n - 1] * f[j2][m + 1] * f[k2][n] * f[i][n - 1];
					E5j1k1df -= 0.5 * (J[k1] * J[j1] / eps(U, j1, k1, m, n))
						* g(m, n) * g(n - 1, m + 1) * ~f[j1][m] * ~f[k1][n - 1]
						* f[j1][m] * f[k1][n] * f[i][n - 1];
					E5j2k2df -= 0.5 * (J[j2] * J[i] / eps(U, j2, k2, m, n))
						* g(m, n) * g(n - 1, m + 1) * ~f[j2][m] * ~f[k2][n - 1]
						* f[j2][m] * f[k2][n] * f[i][n - 1];
				}
			}
		}
		if (n != m - 1) {
			if (n < nmax) {
				if (m > 0) {
					E5j1k1df += 0.5 * (J[k1] * J[j1] / eps(U, k1, j1, n, m))
						* g(n, m) * g(m - 1, n + 1) * ~f[k1][n + 1]
						* ~f[j1][m - 1] * f[k1][n] * f[j1][m - 1] * f[i][n + 1];
					E5j2k2df += 0.5 * (J[j2] * J[i] / eps(U, k2, j2, n, m))
						* g(n, m) * g(m - 1, n + 1) * ~f[k2][n + 1]
						* ~f[j2][m - 1] * f[k2][n] * f[j2][m - 1] * f[i][n + 1];
					E5j1j2df -= 0.5 * (J[j1] * J[i] / eps(U, i, j1, n, m))
						* g(n, m) * g(m - 1, n + 1) * ~f[j1][m - 1] * ~f[j2][m]
						* f[j1][m] * f[j2][m - 1] * f[i][n];
					E5j1j2df -= 0.5 * (J[i] * J[j1] / eps(U, i, j2, n, m))
						* g(n, m) * g(m - 1, n + 1) * ~f[j2][m - 1] * ~f[j1][m]
						* f[j2][m] * f[j1][m - 1] * f[i][n];
				}
				E5j1k1df -= 0.5 * (J[k1] * J[j1] / eps(U, k1, j1, n, m))
					* g(n, m) * g(m - 1, n + 1) * ~f[k1][n + 1] * ~f[j1][m]
					* f[k1][n] * f[j1][m] * f[i][n + 1];
				E5j2k2df -= 0.5 * (J[j2] * J[i] / eps(U, k2, j2, n, m))
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

__global__ void norm2ker(real* x, real* norm2s) {
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
	printf("norm[%d] = %f\n", i, norm2s[i]);
}

void energy(real* x, real* f_dev, real* g_dev, real* U, real* J, real mu,
	real* norm2s) {
//	real* x_host = new real[2*L*dim];
//	memCopy(x_host, x, 2*L*dim*sizeof(real), cudaMemcpyDeviceToHost);
//	for(int j = 0; j < L*dim; j++) {
//		printf("%f %f ", x_host[2*j], x_host[2*j+1]);
//	}
//	printf("\n");

	doublecomplex* E_dev;
	memAlloc<doublecomplex>(&E_dev, 1);
	cudaMemset(E_dev, 0, sizeof(doublecomplex));

	Estruct Es;
	memAlloc<doublecomplex>(&Es.E0, L);
	memAlloc<doublecomplex>(&Es.E1j1, L);
	memAlloc<doublecomplex>(&Es.E1j2, L);
	memAlloc<doublecomplex>(&Es.E2j1, L);
	memAlloc<doublecomplex>(&Es.E2j2, L);
	memAlloc<doublecomplex>(&Es.E3j1, L);
	memAlloc<doublecomplex>(&Es.E3j2, L);
	memAlloc<doublecomplex>(&Es.E4j1j2, L);
	memAlloc<doublecomplex>(&Es.E4j1k1, L);
	memAlloc<doublecomplex>(&Es.E4j2k2, L);
	memAlloc<doublecomplex>(&Es.E5j1j2, L);
	memAlloc<doublecomplex>(&Es.E5j1k1, L);
	memAlloc<doublecomplex>(&Es.E5j2k2, L);
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

	Edfstruct Edfs;
	memAlloc<doublecomplex*>(&Edfs.E0df, L);
	memAlloc<doublecomplex*>(&Edfs.E1j1df, L);
	memAlloc<doublecomplex*>(&Edfs.E1j2df, L);
	doublecomplex* E0df_host[dim];
	doublecomplex* E1j1df_host[dim];
	doublecomplex* E1j2df_host[dim];
	for (int i = 0; i < L; i++) {
		memAlloc<doublecomplex>(&E0df_host[i], dim);
		memAlloc<doublecomplex>(&E1j1df_host[i], dim);
		memAlloc<doublecomplex>(&E1j2df_host[i], dim);
	}
	memCopy(Edfs.E0df, E0df_host, dim * sizeof(doublecomplex),
		cudaMemcpyHostToDevice);
	memCopy(Edfs.E1j1df, E1j1df_host, dim * sizeof(doublecomplex),
		cudaMemcpyHostToDevice);
	memCopy(Edfs.E1j2df, E1j2df_host, dim * sizeof(doublecomplex),
		cudaMemcpyHostToDevice);

	real* norm2_dev;
	memAlloc<real>(&norm2_dev, L);

	norm2ker<<<1, L>>>(x, norm2_dev);
	testKer<<<dim, L>>>(x, norm2_dev, U, J, mu, E_dev, Es);
	testKer<<<dim, L>>>(x, norm2_dev, g_dev, U, J, mu, E_dev, Es, Edfs);

	doublecomplex E;
	memCopy(&E, E_dev, sizeof(doublecomplex), cudaMemcpyDeviceToHost);
	doublecomplex E0[L];
	doublecomplex E1j1[L];
	doublecomplex E1j2[L];
	memCopy(E0, Es.E0, L * sizeof(doublecomplex), cudaMemcpyDeviceToHost);
	memCopy(E1j1, Es.E1j1, L * sizeof(doublecomplex), cudaMemcpyDeviceToHost);
	memCopy(E1j2, Es.E1j2, L * sizeof(doublecomplex), cudaMemcpyDeviceToHost);
	printf("E = %f, %f\n", E.real(), E.imag());
	real* g_host = new real[2 * L * dim];
	memCopy(g_host, g_dev, 2 * L * dim * sizeof(real), cudaMemcpyDeviceToHost);
	printf("g: ");
	for (int i = 0; i < L; i++) {
		for (int n = 0; n <= nmax; n++) {
			printf("%f %f ", g_host[2 * (i * dim + n)],
				g_host[2 * (i * dim + n) + 1]);
		}
//		printf("E0[%d] = %f,%f\tE1j1[%d] = %f,%f\tE1j2[%d] = %f,%f\n", i,
//			E0[i].real(), E0[i].imag(), i, E1j1[i].real(), E1j1[i].imag(), i,
//			E1j2[i].real(), E1j2[i].imag());
	}
	printf("\n");

	memFree(E_dev);
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
	for (int i = 0; i < L; i++) {
		memFree(E0df_host[i]);
		memFree(E1j1df_host[i]);
		memFree(E1j2df_host[i]);
	}
	memFree(Edfs.E0df);
	memFree(Edfs.E1j1df);
	memFree(Edfs.E1j2df);
	memFree(norm2_dev);
//	doublecomplex* E0;
//	doublecomplex* E1j1;
//	doublecomplex* E1j2;
//	memAlloc<doublecomplex>(&E0, L);
//	memAlloc<doublecomplex>(&E1j1, L);
//	memAlloc<doublecomplex>(&E1j2, L);
//	energyfKer<<<L, 1>>>(x, E0, E1j1, E1j2, U, J, mu, norm2s);
//	energygKer<<<L, 1>>>(x, f_dev, g_dev, E0, E1j1, E1j2, U, J, mu, norm2s);
//	memFree(E0);
//	memFree(E1j1);
//	memFree(E1j2);
//	energyKer<<<L, 1>>>(x, f_dev, g_dev, U, J, mu, norm2s);
}

#ifdef UNDEF
__global__ void Efuncker(unsigned ndim, const double *x, double* fval,
	double *grad, void *data) {

	const int i = threadIdx.x;
	if (i >= L) {
		return;
	}

	device_parameters* parms = static_cast<device_parameters*>(data);
	double* U = parms->U;
	double mu = parms->mu;
	double* J = parms->J;

	doublecomplex Ec = doublecomplex::zero();

	const doublecomplex * f[L];
	for (int i = 0; i < L; i++) {
		f[i] = reinterpret_cast<const doublecomplex*>(&x[2 * i * dim]);
	}

//	for (int i = 0; i < L; i++) {
	int j1 = mod(i - 1);
	int j2 = mod(i + 1);
	int k1 = mod(i - 2);
	int k2 = mod(i + 2);
	for (int n = 0; n <= nmax; n++) {
		int k = i * dim + n;
		int l1 = j1 * dim + n;
		int l2 = j2 * dim + n;

		Ec = Ec + (0.5 * U[i] * n * (n - 1) - mu * n) * ~f[i][n] * f[i][n];

		if (n < nmax) {
			Ec = Ec
			+ -J[j1] * g(n, n + 1) * ~f[i][n + 1] * ~f[j1][n] * f[i][n]
			* f[j1][n + 1];
			Ec = Ec
			+ -J[i] * g(n, n + 1) * ~f[i][n + 1] * ~f[j2][n] * f[i][n]
			* f[j2][n + 1];

			if (n > 0) {
				Ec =
				Ec
				+ 0.5 * J[j1] * J[j1] * g(n, n) * g(n - 1, n + 1)
				* ~f[i][n + 1] * ~f[j1][n - 1] * f[i][n - 1]
				* f[j1][n + 1]
				* (1 / eps(U, i, j1, n, n)
					- 1 / eps(U, i, j1, n - 1, n + 1));
				Ec =
				Ec
				+ 0.5 * J[i] * J[i] * g(n, n) * g(n - 1, n + 1)
				* ~f[i][n + 1] * ~f[j2][n - 1] * f[i][n - 1]
				* f[j2][n + 1]
				* (1 / eps(U, i, j2, n, n)
					- 1 / eps(U, i, j2, n - 1, n + 1));
			}

			for (int m = 1; m <= nmax; m++) {
				if (n != m - 1) {
					Ec = Ec
					+ 0.5 * (J[j1] * J[j1] / eps(U, i, j1, n, m)) * g(n, m)
					* g(m - 1, n + 1)
					* (~f[i][n + 1] * ~f[j1][m - 1] * f[i][n + 1]
						* f[j1][m - 1]
						- ~f[i][n] * ~f[j1][m] * f[i][n] * f[j1][m]);
					Ec = Ec
					+ 0.5 * (J[i] * J[i] / eps(U, i, j2, n, m)) * g(n, m)
					* g(m - 1, n + 1)
					* (~f[i][n + 1] * ~f[j2][m - 1] * f[i][n + 1]
						* f[j2][m - 1]
						- ~f[i][n] * ~f[j2][m] * f[i][n] * f[j2][m]);
				}
			}

			if (n > 0) {
				Ec = Ec
				+ 0.5 * (J[j1] * J[i] / eps(U, i, j1, n, n)) * g(n, n)
				* g(n - 1, n + 1) * ~f[i][n + 1] * ~f[j1][n - 1] * ~f[j2][n]
				* f[i][n - 1] * f[j1][n] * f[j2][n + 1];
				Ec = Ec
				+ 0.5 * (J[i] * J[j1] / eps(U, i, j2, n, n)) * g(n, n)
				* g(n - 1, n + 1) * ~f[i][n + 1] * ~f[j2][n - 1] * ~f[j1][n]
				* f[i][n - 1] * f[j2][n] * f[j1][n + 1];
				Ec = Ec
				+ 0.5 * (J[j1] * J[k1] / eps(U, i, j1, n, n)) * g(n, n)
				* g(n - 1, n + 1) * ~f[i][n + 1] * ~f[j1][n - 1] * ~f[k1][n]
				* f[i][n] * f[j1][n + 1] * f[k1][n - 1];
				Ec = Ec
				+ 0.5 * (J[i] * J[j2] / eps(U, i, j2, n, n)) * g(n, n)
				* g(n - 1, n + 1) * ~f[i][n + 1] * ~f[j2][n - 1] * ~f[k2][n]
				* f[i][n] * f[j2][n + 1] * f[k2][n - 1];
				Ec = Ec
				- 0.5 * (J[j1] * J[i] / eps(U, i, j1, n - 1, n + 1)) * g(n, n)
				* g(n - 1, n + 1) * ~f[i][n + 1] * ~f[j1][n] * ~f[j2][n - 1]
				* f[i][n - 1] * f[j1][n + 1] * f[j2][n];
				Ec = Ec
				- 0.5 * (J[i] * J[j1] / eps(U, i, j2, n - 1, n + 1)) * g(n, n)
				* g(n - 1, n + 1) * ~f[i][n + 1] * ~f[j2][n] * ~f[j1][n - 1]
				* f[i][n - 1] * f[j2][n + 1] * f[j1][n];
				Ec = Ec
				- 0.5 * (J[j1] * J[k1] / eps(U, i, j1, n - 1, n + 1)) * g(n, n)
				* g(n - 1, n + 1) * ~f[i][n] * ~f[j1][n - 1] * ~f[k1][n + 1]
				* f[i][n - 1] * f[j1][n + 1] * f[k1][n];
				Ec = Ec
				- 0.5 * (J[i] * J[j2] / eps(U, i, j2, n - 1, n + 1)) * g(n, n)
				* g(n - 1, n + 1) * ~f[i][n] * ~f[j2][n - 1] * ~f[k2][n + 1]
				* f[i][n - 1] * f[j2][n + 1] * f[k2][n];
			}

			for (int m = 1; m <= nmax; m++) {
				if (n != m - 1 && n < nmax) {
					Ec = Ec
					+ 0.5 * (J[j1] * J[i] / eps(U, i, j1, n, m)) * g(n, m)
					* g(m - 1, n + 1) * ~f[i][n + 1] * ~f[j1][m - 1]
					* ~f[j2][m] * f[i][n + 1] * f[j1][m] * f[j2][m - 1];
					Ec = Ec
					+ 0.5 * (J[i] * J[j1] / eps(U, i, j2, n, m)) * g(n, m)
					* g(m - 1, n + 1) * ~f[i][n + 1] * ~f[j2][m - 1]
					* ~f[j1][m] * f[i][n + 1] * f[j2][m] * f[j1][m - 1];
					Ec = Ec
					+ 0.5 * (J[j1] * J[k1] / eps(U, i, j1, n, m)) * g(n, m)
					* g(m - 1, n + 1) * ~f[i][n + 1] * ~f[j1][m - 1]
					* ~f[k1][n] * f[i][n] * f[j1][m - 1] * f[k1][n + 1];
					Ec = Ec
					+ 0.5 * (J[i] * J[j2] / eps(U, i, j2, n, m)) * g(n, m)
					* g(m - 1, n + 1) * ~f[i][n + 1] * ~f[j2][m - 1]
					* ~f[k2][n] * f[i][n] * f[j2][m - 1] * f[k2][n + 1];
					Ec = Ec
					- 0.5 * (J[j1] * J[i] / eps(U, i, j1, n, m)) * g(n, m)
					* g(m - 1, n + 1) * ~f[i][n] * ~f[j1][m - 1] * ~f[j2][m]
					* f[i][n] * f[j1][m] * f[j2][m - 1];
					Ec = Ec
					- 0.5 * (J[i] * J[j1] / eps(U, i, j2, n, m)) * g(n, m)
					* g(m - 1, n + 1) * ~f[i][n] * ~f[j2][m - 1] * ~f[j1][m]
					* f[i][n] * f[j2][m] * f[j1][m - 1];
					Ec = Ec
					- 0.5 * (J[j1] * J[k1] / eps(U, i, j1, n, m)) * g(n, m)
					* g(m - 1, n + 1) * ~f[i][n + 1] * ~f[j1][m] * ~f[k1][n]
					* f[i][n] * f[j1][m] * f[k1][n + 1];
					Ec = Ec
					- 0.5 * (J[i] * J[j2] / eps(U, i, j2, n, m)) * g(n, m)
					* g(m - 1, n + 1) * ~f[i][n + 1] * ~f[j2][m] * ~f[k2][n]
					* f[i][n] * f[j2][m] * f[k2][n + 1];
				}
			}

		}
	}

	if (grad) {
//		for (int i = 0; i < L; i++) {
		int j1 = mod(i - 1);
		int j2 = mod(i + 1);
		int k1 = mod(i - 2);
		int k2 = mod(i + 2);
		for (int n = 0; n <= nmax; n++) {
			int k = i * dim + n;
			int l1 = j1 * dim + n;
			int l2 = j2 * dim + n;

			doublecomplex dfc = doublecomplex::zero();

			dfc = dfc + (U[i] * n * (n - 1) - 2 * mu * n) * f[i][n];

			if (n < nmax) {
				dfc = dfc
				+ -2 * J[j1] * g(n, n + 1) * ~f[j1][n + 1] * f[j1][n]
				* f[i][n + 1];
				dfc = dfc
				+ -2 * J[i] * g(n, n + 1) * ~f[j2][n + 1] * f[j2][n]
				* f[i][n + 1];
			}
			if (n > 0) {
				dfc = dfc
				+ -2 * J[j1] * g(n - 1, n) * ~f[j1][n - 1] * f[j1][n]
				* f[i][n - 1];
				dfc = dfc
				+ -2 * J[i] * g(n - 1, n) * ~f[j2][n - 1] * f[j2][n]
				* f[i][n - 1];
			}

			if (n > 1) {
				dfc = dfc
				+ J[j1] * J[j1] * g(n - 1, n - 1) * g(n - 2, n) * ~f[j1][n - 2]
				* f[j1][n] * f[i][n - 2]
				* (1 / eps(U, i, j1, n - 1, n - 1)
					- 1 / eps(U, i, j1, n - 2, n));
				dfc = dfc
				+ J[i] * J[i] * g(n - 1, n - 1) * g(n - 2, n) * ~f[j2][n - 2]
				* f[j2][n] * f[i][n - 2]
				* (1 / eps(U, i, j2, n - 1, n - 1)
					- 1 / eps(U, i, j2, n - 2, n));
			}
			if (n < nmax - 1) {
				dfc = dfc
				+ J[j1] * J[j1] * g(n + 1, n + 1) * g(n, n + 2) * ~f[j1][n + 2]
				* f[j1][n] * f[i][n + 2]
				* (1 / eps(U, j1, i, n + 1, n + 1)
					- 1 / eps(U, j1, i, n, n + 2));
				dfc = dfc
				+ J[i] * J[i] * g(n + 1, n + 1) * g(n, n + 2) * ~f[j2][n + 2]
				* f[j2][n] * f[i][n + 2]
				* (1 / eps(U, j2, i, n + 1, n + 1)
					- 1 / eps(U, j2, i, n, n + 2));
			}

			for (int m = 0; m < nmax; m++) {
				if (n != m + 1) {
					dfc = dfc
					+ (1 / eps(U, i, j1, n - 1, m + 1)) * J[j1] * J[j1]
					* g(n - 1, m + 1) * g(m, n) * ~f[j1][m] * f[j1][m]
					* f[i][n];
					dfc = dfc
					+ (1 / eps(U, i, j2, n - 1, m + 1)) * J[i] * J[i]
					* g(n - 1, m + 1) * g(m, n) * ~f[j2][m] * f[j2][m]
					* f[i][n];
					dfc = dfc
					- (1 / eps(U, j1, i, m, n)) * J[j1] * J[j1] * g(m, n)
					* g(n - 1, m + 1) * ~f[j1][m] * f[j1][m] * f[i][n];
					dfc = dfc
					- (1 / eps(U, j2, i, m, n)) * J[i] * J[i] * g(m, n)
					* g(n - 1, m + 1) * ~f[j2][m] * f[j2][m] * f[i][n];
				}
				if (n != m && n < nmax) {
					dfc = dfc
					+ (1 / eps(U, j1, i, m, n + 1)) * J[j1] * J[j1]
					* g(m, n + 1) * g(n, m + 1) * ~f[j1][m + 1]
					* f[j1][m + 1] * f[i][n];
					dfc = dfc
					+ (1 / eps(U, j2, i, m, n + 1)) * J[i] * J[i] * g(m, n + 1)
					* g(n, m + 1) * ~f[j2][m + 1] * f[j2][m + 1] * f[i][n];
					dfc = dfc
					- (1 / eps(U, i, j1, n, m + 1)) * J[j1] * J[j1]
					* g(n, m + 1) * g(m, n + 1) * ~f[j1][m + 1]
					* f[j1][m + 1] * f[i][n];
					dfc = dfc
					- (1 / eps(U, i, j2, n, m + 1)) * J[i] * J[i] * g(n, m + 1)
					* g(m, n + 1) * ~f[j2][m + 1] * f[j2][m + 1] * f[i][n];
				}
			}

			if (n >= 2) {
				dfc = dfc
				+ (1 / eps(U, i, j1, n - 1, n - 1)) * g(n - 1, n - 1)
				* g(n - 2, n) * J[j1] * J[i] * ~f[j1][n - 2] * ~f[j2][n - 1]
				* f[j1][n - 1] * f[j2][n] * f[i][n - 2];
				dfc = dfc
				+ (1 / eps(U, i, j2, n - 1, n - 1)) * g(n - 1, n - 1)
				* g(n - 2, n) * J[i] * J[j1] * ~f[j2][n - 2] * ~f[j1][n - 1]
				* f[j2][n - 1] * f[j1][n] * f[i][n - 2];
				dfc = dfc
				+ (1 / eps(U, i, j1, n - 1, n - 1)) * g(n - 1, n - 1)
				* g(n - 2, n) * J[j1] * J[k1] * ~f[j1][n - 2]
				* ~f[k1][n - 1] * f[j1][n] * f[k1][n - 2] * f[i][n - 1];
				dfc = dfc
				+ (1 / eps(U, i, j2, n - 1, n - 1)) * g(n - 1, n - 1)
				* g(n - 2, n) * J[i] * J[j2] * ~f[j2][n - 2] * ~f[k2][n - 1]
				* f[j2][n] * f[k2][n - 2] * f[i][n - 1];
				dfc = dfc
				- (1 / eps(U, i, j1, n - 2, n)) * g(n - 1, n - 1) * g(n - 2, n)
				* J[j1] * J[i] * ~f[j1][n - 1] * ~f[j2][n - 2] * f[j1][n]
				* f[j2][n - 1] * f[i][n - 2];
				dfc = dfc
				- (1 / eps(U, i, j2, n - 2, n)) * g(n - 1, n - 1) * g(n - 2, n)
				* J[i] * J[j1] * ~f[j2][n - 1] * ~f[j1][n - 2] * f[j2][n]
				* f[j1][n - 1] * f[i][n - 2];
				dfc = dfc
				- (1 / eps(U, k1, j1, n - 2, n)) * g(n - 1, n - 1) * g(n - 2, n)
				* J[k1] * J[j1] * ~f[k1][n - 1] * ~f[j1][n - 2]
				* f[k1][n - 2] * f[j1][n] * f[i][n - 1];
				dfc = dfc
				- (1 / eps(U, k2, j2, n - 2, n)) * g(n - 1, n - 1) * g(n - 2, n)
				* J[j2] * J[i] * ~f[k2][n - 1] * ~f[j2][n - 2]
				* f[k2][n - 2] * f[j2][n] * f[i][n - 1];
			}
			if (n >= 1 && n <= nmax - 1) {
				dfc = dfc
				+ (1 / eps(U, j1, k1, n, n)) * g(n, n) * g(n - 1, n + 1) * J[k1]
				* J[j1] * ~f[j1][n + 1] * ~f[k1][n - 1] * f[j1][n - 1]
				* f[k1][n] * f[i][n + 1];
				dfc = dfc
				+ (1 / eps(U, j2, k2, n, n)) * g(n, n) * g(n - 1, n + 1) * J[j2]
				* J[i] * ~f[j2][n + 1] * ~f[k2][n - 1] * f[j2][n - 1]
				* f[k2][n] * f[i][n + 1];
				dfc = dfc
				+ (1 / eps(U, k1, j1, n, n)) * g(n, n) * g(n - 1, n + 1) * J[k1]
				* J[j1] * ~f[k1][n + 1] * ~f[j1][n - 1] * f[k1][n]
				* f[j1][n + 1] * f[i][n - 1];
				dfc = dfc
				+ (1 / eps(U, k2, j2, n, n)) * g(n, n) * g(n - 1, n + 1) * J[j2]
				* J[i] * ~f[k2][n + 1] * ~f[j2][n - 1] * f[k2][n]
				* f[j2][n + 1] * f[i][n - 1];
				dfc = dfc
				- (1 / eps(U, j1, i, n - 1, n + 1)) * g(n, n) * g(n - 1, n + 1)
				* J[j1] * J[k1] * ~f[j1][n + 1] * ~f[k1][n - 1]
				* f[j1][n - 1] * f[k1][n] * f[i][n + 1];
				dfc = dfc
				- (1 / eps(U, j2, i, n - 1, n + 1)) * g(n, n) * g(n - 1, n + 1)
				* J[i] * J[j2] * ~f[j2][n + 1] * ~f[k2][n - 1]
				* f[j2][n - 1] * f[k2][n] * f[i][n + 1];
				dfc = dfc
				- (1 / eps(U, i, j1, n - 1, n + 1)) * g(n, n) * g(n - 1, n + 1)
				* J[j1] * J[k1] * ~f[j1][n - 1] * ~f[k1][n + 1]
				* f[j1][n + 1] * f[k1][n] * f[i][n - 1];
				dfc = dfc
				- (1 / eps(U, i, j2, n - 1, n + 1)) * g(n, n) * g(n - 1, n + 1)
				* J[i] * J[j2] * ~f[j2][n - 1] * ~f[k2][n + 1]
				* f[j2][n + 1] * f[k2][n] * f[i][n - 1];
			}
			if (n <= nmax - 2) {
				dfc = dfc
				+ (1 / eps(U, j1, i, n + 1, n + 1)) * g(n + 1, n + 1)
				* g(n, n + 2) * J[j1] * J[k1] * ~f[j1][n + 2]
				* ~f[k1][n + 1] * f[j1][n] * f[k1][n + 2] * f[i][n + 1];
				dfc = dfc
				+ (1 / eps(U, j2, i, n + 1, n + 1)) * g(n + 1, n + 1)
				* g(n, n + 2) * J[i] * J[j2] * ~f[j2][n + 2] * ~f[k2][n + 1]
				* f[j2][n] * f[k2][n + 2] * f[i][n + 1];
				dfc = dfc
				+ (1 / eps(U, j1, i, n + 1, n + 1)) * g(n + 1, n + 1)
				* g(n, n + 2) * J[j1] * J[i] * ~f[j1][n + 2] * ~f[j2][n + 1]
				* f[j1][n + 1] * f[j2][n] * f[i][n + 2];
				dfc = dfc
				+ (1 / eps(U, j2, i, n + 1, n + 1)) * g(n + 1, n + 1)
				* g(n, n + 2) * J[i] * J[j1] * ~f[j2][n + 2] * ~f[j1][n + 1]
				* f[j2][n + 1] * f[j1][n] * f[i][n + 2];
				dfc = dfc
				- (1 / eps(U, j1, k1, n, n + 2)) * g(n + 1, n + 1) * g(n, n + 2)
				* J[k1] * J[j1] * ~f[j1][n + 2] * ~f[k1][n + 1] * f[j1][n]
				* f[k1][n + 2] * f[i][n + 1];
				dfc = dfc
				- (1 / eps(U, j2, k2, n, n + 2)) * g(n + 1, n + 1) * g(n, n + 2)
				* J[j2] * J[i] * ~f[j2][n + 2] * ~f[k2][n + 1] * f[j2][n]
				* f[k2][n + 2] * f[i][n + 1];
				dfc = dfc
				- (1 / eps(U, j1, i, n, n + 2)) * g(n + 1, n + 1) * g(n, n + 2)
				* J[j1] * J[i] * ~f[j1][n + 1] * ~f[j2][n + 2] * f[j1][n]
				* f[j2][n + 1] * f[i][n + 2];
				dfc = dfc
				- (1 / eps(U, j2, i, n, n + 2)) * g(n + 1, n + 1) * g(n, n + 2)
				* J[i] * J[j1] * ~f[j2][n + 1] * ~f[j1][n + 2] * f[j2][n]
				* f[j1][n + 1] * f[i][n + 2];
			}

			for (int m = 0; m <= nmax; m++) {
				if (n != m) {
					if (n > 0) {
						if (m > 0) {
							dfc = dfc
							+ (1 / eps(U, i, j1, n - 1, m)) * J[j1] * J[i]
							* g(n - 1, m) * g(m - 1, n) * ~f[j1][m - 1]
							* ~f[j2][m] * f[j1][m] * f[j2][m - 1] * f[i][n];
							dfc = dfc
							+ (1 / eps(U, i, j2, n - 1, m)) * J[i] * J[j1]
							* g(n - 1, m) * g(m - 1, n) * ~f[j2][m - 1]
							* ~f[j1][m] * f[j2][m] * f[j1][m - 1] * f[i][n];
							dfc = dfc
							+ (1 / eps(U, i, j1, n - 1, m)) * J[j1] * J[k1]
							* g(n - 1, m) * g(m - 1, n) * ~f[j1][m - 1]
							* ~f[k1][n - 1] * f[j1][m - 1] * f[k1][n]
							* f[i][n - 1];
							dfc = dfc
							+ (1 / eps(U, i, j2, n - 1, m)) * J[i] * J[j2]
							* g(n - 1, m) * g(m - 1, n) * ~f[j2][m - 1]
							* ~f[k2][n - 1] * f[j2][m - 1] * f[k2][n]
							* f[i][n - 1];
							dfc = dfc
							- (1 / eps(U, i, j1, n - 1, m)) * J[j1] * J[k1]
							* g(n - 1, m) * g(m - 1, n) * ~f[j1][m]
							* ~f[k1][n - 1] * f[j1][m] * f[k1][n]
							* f[i][n - 1];
							dfc = dfc
							- (1 / eps(U, i, j2, n - 1, m)) * J[i] * J[j2]
							* g(n - 1, m) * g(m - 1, n) * ~f[j2][m]
							* ~f[k2][n - 1] * f[j2][m] * f[k2][n]
							* f[i][n - 1];
						}
					}
					if (n < nmax) {
						if (m < nmax) {
							dfc = dfc
							+ (1 / eps(U, j1, i, m, n + 1)) * J[j1] * J[k1]
							* g(m, n + 1) * g(n, m + 1) * ~f[j1][m + 1]
							* ~f[k1][n + 1] * f[j1][m + 1] * f[k1][n]
							* f[i][n + 1];
							dfc = dfc
							+ (1 / eps(U, j2, i, m, n + 1)) * J[i] * J[j2]
							* g(m, n + 1) * g(n, m + 1) * ~f[j2][m + 1]
							* ~f[k2][n + 1] * f[j2][m + 1] * f[k2][n]
							* f[i][n + 1];
							dfc = dfc
							+ (1 / eps(U, j1, i, m, n + 1)) * J[j1] * J[i]
							* g(m, n + 1) * g(n, m + 1) * ~f[j1][m + 1]
							* ~f[j2][m] * f[j1][m] * f[j2][m + 1] * f[i][n];
							dfc = dfc
							+ (1 / eps(U, j2, i, m, n + 1)) * J[i] * J[j1]
							* g(m, n + 1) * g(n, m + 1) * ~f[j2][m + 1]
							* ~f[j1][m] * f[j2][m] * f[j1][m + 1] * f[i][n];
							dfc = dfc
							- (1 / eps(U, j1, i, m, n + 1)) * J[j1] * J[k1]
							* g(m, n + 1) * g(n, m + 1) * ~f[j1][m]
							* ~f[k1][n + 1] * f[j1][m] * f[k1][n]
							* f[i][n + 1];
							dfc = dfc
							- (1 / eps(U, j2, i, m, n + 1)) * J[i] * J[j2]
							* g(m, n + 1) * g(n, m + 1) * ~f[j2][m]
							* ~f[k2][n + 1] * f[j2][m] * f[k2][n]
							* f[i][n + 1];
						}
					}
				}
				if (n != m + 1) {
					if (m < nmax) {
						dfc = dfc
						- (1 / eps(U, j1, i, m, n)) * J[j1] * J[i] * g(m, n)
						* g(n - 1, m + 1) * ~f[j1][m + 1] * ~f[j2][m]
						* f[j1][m] * f[j2][m + 1] * f[i][n];
						dfc = dfc
						- (1 / eps(U, j2, i, m, n)) * J[i] * J[j1] * g(m, n)
						* g(n - 1, m + 1) * ~f[j2][m + 1] * ~f[j1][m]
						* f[j2][m] * f[j1][m + 1] * f[i][n];
					}
					if (n > 0) {
						if (m < nmax) {
							dfc = dfc
							+ (1 / eps(U, j1, k1, m, n)) * J[k1] * J[j1]
							* g(m, n) * g(n - 1, m + 1) * ~f[j1][m + 1]
							* ~f[k1][n - 1] * f[j1][m + 1] * f[k1][n]
							* f[i][n - 1];
							dfc = dfc
							+ (1 / eps(U, j2, k2, m, n)) * J[j2] * J[i]
							* g(m, n) * g(n - 1, m + 1) * ~f[j2][m + 1]
							* ~f[k2][n - 1] * f[j2][m + 1] * f[k2][n]
							* f[i][n - 1];
							dfc = dfc
							- (1 / eps(U, j1, k1, m, n)) * J[k1] * J[j1]
							* g(m, n) * g(n - 1, m + 1) * ~f[j1][m]
							* ~f[k1][n - 1] * f[j1][m] * f[k1][n]
							* f[i][n - 1];
							dfc = dfc
							- (1 / eps(U, j2, k2, m, n)) * J[j2] * J[i]
							* g(m, n) * g(n - 1, m + 1) * ~f[j2][m]
							* ~f[k2][n - 1] * f[j2][m] * f[k2][n]
							* f[i][n - 1];
						}
					}
				}
				if (n != m - 1) {
					if (n < nmax) {
						if (m > 0) {
							dfc = dfc
							+ (1 / eps(U, k1, j1, n, m)) * J[k1] * J[j1]
							* g(n, m) * g(m - 1, n + 1) * ~f[k1][n + 1]
							* ~f[j1][m - 1] * f[k1][n] * f[j1][m - 1]
							* f[i][n + 1];
							dfc = dfc
							+ (1 / eps(U, k2, j2, n, m)) * J[j2] * J[i]
							* g(n, m) * g(m - 1, n + 1) * ~f[k2][n + 1]
							* ~f[j2][m - 1] * f[k2][n] * f[j2][m - 1]
							* f[i][n + 1];
							dfc = dfc
							- (1 / eps(U, i, j1, n, m)) * J[j1] * J[i] * g(n, m)
							* g(m - 1, n + 1) * ~f[j1][m - 1] * ~f[j2][m]
							* f[j1][m] * f[j2][m - 1] * f[i][n];
							dfc = dfc
							- (1 / eps(U, i, j2, n, m)) * J[i] * J[j1] * g(n, m)
							* g(m - 1, n + 1) * ~f[j2][m - 1] * ~f[j1][m]
							* f[j2][m] * f[j1][m - 1] * f[i][n];
						}
						dfc = dfc
						- (1 / eps(U, k1, j1, n, m)) * J[k1] * J[j1] * g(n, m)
						* g(m - 1, n + 1) * ~f[k1][n + 1] * ~f[j1][m]
						* f[k1][n] * f[j1][m] * f[i][n + 1];
						dfc = dfc
						- (1 / eps(U, k2, j2, n, m)) * J[j2] * J[i] * g(n, m)
						* g(m - 1, n + 1) * ~f[k2][n + 1] * ~f[j2][m]
						* f[k2][n] * f[j2][m] * f[i][n + 1];
					}
				}
			}

			grad[2 * k] = dfc.real();
			grad[2 * k + 1] = dfc.imag();
		}
//		}
	}

//	printf("E: %f\n", Ec.real());
	atomicAdd(fval, Ec.real());
//	fval[0] = Ec.real();

//    cout << Ec.real() << endl;
//	return Ec.real();
}

#endif

