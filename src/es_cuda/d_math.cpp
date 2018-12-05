#include "d_math.h"
#include <stdio.h>

#define BLOCK_SIZE 32

dim3 dimGrid(int m, int k) {
	return dim3((k + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);
}

dim3 dimBlock() {
	return dim3(BLOCK_SIZE, BLOCK_SIZE);
}

__global__ void addKernel(float *c, const float *a, const float *b) {
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

__global__ void subtractKernel(float *c, const float *a, const float *b) {
	int i = threadIdx.x;
	c[i] = a[i] - b[i];
}

__global__ void MatrixMultKernel(float *dst,const float *srcA,const float *srcB, int m, int n, int k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(col < k && row < m) {
		float tempSum = 0.f;
		for(int ind = 0; ind < n; ++ind) {
			tempSum += srcA[row + m * ind] * srcB[col * n + ind];
		}
		dst[col * m + row] = tempSum;
	}
}

__global__ void ForwardLayerKernel(float *dst,const float *d_W, const float *d_last, const float * d_bias, int m, int n, int k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(col < k && row < m) {
		float tempSum = 0.f;
		for(int ind = 0; ind < n; ++ind) {
			tempSum += d_W[row + m * ind] * d_last[col * n + ind];
		}
		dst[col * m + row] = tempSum + d_bias[row];
	}
}


void d_forwardLayer(float* dst, const float* d_W, const float* d_last, const float* d_bias, int  m, int n, int k) {	
	ForwardLayerKernel << <dimGrid(m,k), dimBlock() >> > 
		(dst, d_W, d_last, d_bias, m, n, k);
}

void d_matrixMult(float* dst, const float* d_W, const float* d_last, int  m, int n, int k) {
	MatrixMultKernel << <dimGrid(m, k), dimBlock() >> > 
		(dst, d_W, d_last, m, n, k);
}

__global__ void SigmoidKernal(float *dst, int N) {
	int tid = blockIdx.x;
	if(tid < N)
		dst[tid] = 1/(1+exp(-dst[tid]));
}

__global__ void TanhKernal(float *dst, int N) {
		int tid = blockIdx.x;
		if(tid < N)
			dst[tid] = tanh(dst[tid]);
}

__global__ void ReLUKernal(float *dst, int N) {
	int tid = blockIdx.x;
	if(tid < N)
		dst[tid] = max(0.f, dst[tid]);
}

__global__ void LReLUKernal(float *dst, int N) {
	int tid = blockIdx.x;
	if(tid < N)
		dst[tid] = max(dst[tid] * 0.1f, dst[tid]);
}


void d_Activate(float* dst, int size, Activation act) {
	switch(act) { //TODO: set up common types
	case Sigmoid:
		SigmoidKernal << < size, 1 >> > (dst, size);
	case Tanh:
		TanhKernal << < size, 1 >> > (dst, size);
		break;
	case ReLU:
		ReLUKernal<<<size,1>>>(dst, size);
		break;
	case LReLU:
		LReLUKernal << < size, 1>>>(dst, size);
		break;
	case Linear: //fall through
	default:
		break;
	}
}


__global__ void BackTanhKernel(float *dst, float *srcA, float *srcB, const float *d_A, int m, int n, int k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(col < k && row < m) {
		float tempSum = 0.f;
		for(int ind = 0; ind < n; ++ind) {
			tempSum += srcA[row + m * ind] * srcB[col * n + ind];
		}
		dst[col * m + row] = tempSum * (1 - d_A[col * m + row] * d_A[col * m + row]);
	}
}


void d_BackTanh(float *dst, float *d_W, float *d_dZ, const float *d_A, int m, int n, int k) {
	BackTanhKernel << <dimGrid(m,k), dimBlock() >> > (dst, d_W, d_dZ, d_A, m, n, k);
}

__global__ void BackReLUKernel(float *dst, float *srcA, float *srcB, const float *d_A, int m, int n, int k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(col < k && row < m) {
		float tempSum = 0.f;
		for(int ind = 0; ind < n; ++ind) {
			tempSum += srcA[row + m * ind] * srcB[col * n + ind];
		}
		dst[col * m + row] = tempSum *(d_A[col * m + row] > 0.f ? 1.f : 0.f);
	}
}


void d_BackReLU(float *dst, float *d_W, float *d_dZ, const float *d_A, int m, int n, int k) {
	BackReLUKernel << <dimGrid(m, k), dimBlock() >> > (dst, d_W, d_dZ, d_A, m, n, k);
}

// Helper function for using CUDA to add vectors in parallel.
void d_add(float *c, const float *a, const float *b, unsigned int size) {
	addKernel <<<1,size>>> (c, a, b);
}

// Helper function for using CUDA to add vectors in parallel.
void d_subtract(float *c, const float *a, const float *b, unsigned int size) {
	subtractKernel << <1, size >> > (c, a, b);
}
