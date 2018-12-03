#include "d_math.h"
#include <stdio.h>

#define BLOCK_SIZE 32

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b) {
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

__global__ void MatrixMultKernel(float *dst, float *srcA, float *srcB, int m, int n, int k) {
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

__global__ void ForwardLayerKernel(float *dst, float *d_W, float *d_last, float * d_bias, int m, int n, int k) {
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

void d_forwardLayer(float* dst, float* d_W, float* d_last, float* d_bias, int  m, int n, int k) {
	dim3 dimGrid((k + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	ForwardLayerKernel << <dimGrid, dimBlock >> > (dst, d_W, d_last, d_bias, m, n, k);
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


void d_Activate(float* dst, int size, int act) {
	switch(act) {
	case 1: //Sigmoid
		SigmoidKernal << < size, 1 >> > (dst, size);
	case 2: //Tanh:
		TanhKernal << < size, 1 >> > (dst, size);
		break;
	case 3://ReLU:
		ReLUKernal<<<size,1>>>(dst, size);
		break;
	case 4://LReLU:
		LReLUKernal << < size, 1>>>(dst, size);
		break;
	case 0://Linear: //fall through
	default:
		break;
	}
}


// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size) {
	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if(cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
	if(cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
	if(cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
	if(cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if(cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
	if(cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	addKernel <<<1, size >>> (dev_c, dev_a, dev_b);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if(cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if(cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	if(cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaStatus;
}
