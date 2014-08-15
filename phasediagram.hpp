/*
 * phasediagram.hpp
 *
 *  Created on: Aug 12, 2014
 *      Author: Abuenameh
 */

#ifndef PHASEDIAGRAM_HPP_
#define PHASEDIAGRAM_HPP_

#include "cudacomplex.hpp"

#define L 3
#define nmax 7
#define dim (nmax+1)

__host__ __device__ inline int mod(int i) {
	return (i + L) % L;
}

__host__ __device__ inline double g(int n, int m) {
	return sqrt(1.0 * (n + 1) * m);
}

__host__ __device__ inline double eps(double* U, int i, int j, int n, int m) {
	return n * U[i] - (m - 1) * U[j];
}

// Define this to turn on error checking
#define CUDA_ERROR_CHECK

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError() __cudaCheckError( __FILE__, __LINE__ )
#define CudaSafeMemAllocCall(ret) __cudaSafeMemAllocCall(ret, __FILE__, __LINE__)

inline void __cudaSafeCall(cudaError err, const char *file, const int line) {
#ifdef CUDA_ERROR_CHECK
	if (cudaSuccess != err) {
		fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n", file, line,
			cudaGetErrorString(err));
		exit(-1);
	}
#endif

	return;
}

inline void __cudaCheckError(const char *file, const int line) {
#ifdef CUDA_ERROR_CHECK
	cudaError err = cudaGetLastError();
	if (cudaSuccess != err) {
		fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n", file, line,
			cudaGetErrorString(err));
		exit(-1);
	}

// More careful checking. However, this will affect performance.
// Comment away if needed.
	err = cudaDeviceSynchronize();
	if (cudaSuccess != err) {
		fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
			file, line, cudaGetErrorString(err));
		exit(-1);
	}
#endif

	return;
}

inline void __cudaSafeMemAllocCall(int ret, const char *file, const int line) {
#ifdef CUDA_ERROR_CHECK
	if (ret == -1) {
		__cudaCheckError(file, line);
	}
#endif

	return;
}

struct Parameters {
	real* U;
	real* J;
	real mu;
};

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

struct Workspace {
	real* norm2;
};

void allocE(Estruct** Es_dev);
void freeE(Estruct* Es_dev);

void allocWorkspace(Workspace& work);
void freeWorkspace(Workspace& work);

#endif /* PHASEDIAGRAM_HPP_ */
