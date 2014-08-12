/*
 * functions.cu
 *
 *  Created on: Aug 12, 2014
 *      Author: Abuenameh
 */

#include "L-BFGS/cutil_inline.h"
#include "L-BFGS/lbfgsbcuda.h"
#include "L-BFGS/lbfgsb.h"
#include "cudacomplex.hpp"
#include "phasediagram.hpp"

__global__ void energyKer(real* x, real* f_dev, real* g_dev, real* U,
		real* J, real mu, real* norm2) {

	if(threadIdx.x > 0) {
		return;
	}

	const int i = blockIdx.x;
	if (i >= L) {
		return;
	}

	//	__shared__ doublecomplex f[L*dim];
		__shared__ doublecomplex fi[5*dim];
		__shared__ doublecomplex* f[5];

	int j1 = mod(i - 1);
	int j2 = mod(i + 1);
	int k1 = mod(i - 2);
	int k2 = mod(i + 2);

	f[0] = &fi[k1*dim];
	f[1] = &fi[j1*dim];
	f[2] = &fi[i*dim];
	f[3] = &fi[j2*dim];
	f[4] = &fi[k2*dim];

	for(int n = 0; n <= nmax; n++) {
				f[0][n] = make_doublecomplex(x[2*(k1*dim+n)], x[2*(k1*dim+n)+1]);
				f[1][n] = make_doublecomplex(x[2*(j1*dim+n)], x[2*(j1*dim+n)+1]);
				f[2][n] = make_doublecomplex(x[2*(i*dim+n)], x[2*(i*dim+n)+1]);
				f[3][n] = make_doublecomplex(x[2*(j2*dim+n)], x[2*(j2*dim+n)+1]);
				f[4][n] = make_doublecomplex(x[2*(k2*dim+n)], x[2*(k2*dim+n)+1]);
//		f[i*dim+n] = make_doublecomplex(x[2*(i*dim+n)], x[2*(i*dim+n)+1]);
//		f[j1*dim+n] = make_doublecomplex(x[2*(i*dim+n)], x[2*(i*dim+n)+1]);
//		f[j2*dim+n] = make_doublecomplex(x[2*(i*dim+n)], x[2*(i*dim+n)+1]);
//		f[k1*dim+n] = make_doublecomplex(x[2*(i*dim+n)], x[2*(i*dim+n)+1]);
//		f[k2*dim+n] = make_doublecomplex(x[2*(i*dim+n)], x[2*(i*dim+n)+1]);
	}

//	for(int i = 0; i < L; i++) {
//		for(int n = 0; n <= nmax; n++) {
//			f[i*dim+n] = make_doublecomplex(x[2*(i*dim+n)])
//		}
//	}
//
//	const doublecomplex * f[L];
//	for (int i = 0; i < L; i++) {
//		f[i] = reinterpret_cast<const doublecomplex*>(&x[2 * i * dim]);
//	}

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


