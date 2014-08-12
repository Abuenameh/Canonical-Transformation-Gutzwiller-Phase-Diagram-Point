/*
 * phasediagram.hpp
 *
 *  Created on: Aug 12, 2014
 *      Author: Abuenameh
 */

#ifndef PHASEDIAGRAM_HPP_
#define PHASEDIAGRAM_HPP_

#define L 3
#define nmax 3
#define dim (nmax+1)

__host__ __device__ inline int mod(int i) {
	return (i + L) % L;
}

__host__ __device__ inline double g(int n, int m) {
	return sqrt(1.0*(n + 1) * m);
}

__host__ __device__ inline double eps(double* U, int i, int j, int n, int m) {
	return n * U[i] - (m - 1) * U[j];
}


#endif /* PHASEDIAGRAM_HPP_ */
