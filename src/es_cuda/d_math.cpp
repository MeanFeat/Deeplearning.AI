#include "d_math.h"
#include <stdio.h>


dim3 dimGrid(int m, int k) {
	return dim3((k + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);
}

dim3 dimBlock() {
	return dim3(BLOCK_SIZE, BLOCK_SIZE);
}

__global__ void add_Kernel(double *c, const double *a, const double *b) {
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
} /* dst = srcA (+) srcB */
void d_add(d_Matrix *dst, d_Matrix *srcA, d_Matrix *srcB) {
	add_Kernel << <1, dst->size() >> > (dst->d_data(), srcA->d_data(), srcB->d_data());
	d_catchErr();
}

__global__ void subtract_Kernel(double *c, const double *a, const double *b) {
	int i = threadIdx.x;
	c[i] = a[i] - b[i];
} /* dst = srcA (-) srcB */
void d_subtract(d_Matrix *dst, d_Matrix *srcA, d_Matrix *srcB) {
	subtract_Kernel << <1, dst->size() >> > (dst->d_data(), srcA->d_data(), srcB->d_data());
	d_catchErr();
}

__global__ void mult_elem_Kernel(double *c, double *a, double b) {
	int i = threadIdx.x;
	c[i] = a[i] * b;
}

__global__ void mult_Kernel(double *dst, double *srcA, double *srcB, int m, int n, int k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	double sum = 0.0;
	if(col < k && row < m) {
		for(int i = 0; i < n; i++) {
			sum += srcA[row * n + i] * srcB[i * k + col];
		}
		dst[row * k + col] = sum;
	}
}
void d_mult(d_Matrix* dst, d_Matrix* srcA, d_Matrix* srcB) {
	int m = srcA->rows();
	int n = srcA->cols();
	int k = srcB->cols();
	mult_Kernel << <dimGrid(m, k), dimBlock() >> >
		(dst->d_data(), srcA->d_data(), srcB->d_data(), m, n, k);
	d_catchErr();
}

__global__ void mult_lhsT_Kernel(double *dst, double *srcA, double *srcB, int m, int n, int k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(col < k && row < m) {
		double sum = 0.0;
		for(int i = 0; i < n; ++i) {
			sum += srcA[row + m * i] * srcB[i * k + col];
		}
		dst[row * k + col] = sum;
	}
} /*dst = srcA.T * srcB */
void d_mult_lhsT(d_Matrix* dst, d_Matrix* srcA, d_Matrix* srcB) {
	int m = srcA->cols(); //reverse for transpose
	int n = srcA->rows(); //reverse for transpose
	int k = srcB->cols();
	mult_lhsT_Kernel << <dimGrid(m, k), dimBlock() >> >
		(dst->d_data(), srcA->d_data(), srcB->d_data(), m, n, k);
	d_catchErr();
}

__global__ void mult_rhsT_Kernel(double *dst, const double *srcA, const double *srcB, int m, int n, int k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	double sum = 0.0;
	if(col < k && row < m) {
		for(int i = 0; i < n; i++) {
			sum += srcA[row * n + i] * srcB[col * n + i];
		}
		dst[row * k + col] = sum;
	}
} /* dst = srcA * srcB.T */
void d_mult_rhsT(d_Matrix* dst, d_Matrix* srcA, d_Matrix* srcB) {
	int m = srcA->rows();
	int n = srcA->cols();
	int k = srcB->rows(); //reverse for transpose
	mult_rhsT_Kernel << <dimGrid(m, k), dimBlock() >> >
		(dst->d_data(), srcA->d_data(), srcB->d_data(), m, n, k);
	d_catchErr();
}

__global__ void forwardLayer_Kernel(double *dst,const double *d_W, const double *d_last, const double * d_bias, int m, int n, int k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	double sum = 0.0;
	if(col < k && row < m) {
		for(int i = 0; i < n; i++) {
			sum += d_W[row * n + i] * d_last[i * k + col];
		}
		dst[row * k + col] = sum + d_bias[row];
	}
} /* dst = d_W * d_last + d_bias */
void d_forwardLayer(d_Matrix *dst, d_Matrix *d_W, d_Matrix *d_last, d_Matrix *d_bias) {
	int m = d_W->rows();
	int n = d_W->cols();
	int k = d_last->cols();
	forwardLayer_Kernel << <dimGrid(m, k), dimBlock() >> >
		(dst->d_data(), d_W->d_data(), d_last->d_data(), d_bias->d_data(), m, n, k);
	d_catchErr();
}

__global__ void drawPixels_Kernel(int *buffer, int k, const double* vals, bool discrete, Color neg, Color pos) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	double percent = vals[row * k + col];
	if(discrete) {
		if (percent > 0.0) {
			buffer[row * k + col] = ((pos.r << 16) | ((pos.g << 8) | pos.b));
		} else {
			buffer[row * k + col] = ((neg.r << 16) | ((neg.g << 8) | neg.b));
		}
	} else {
		if(percent > 0.) {
			unsigned char r = unsigned char(double(255) + (percent*(double(pos.r) - double(255))));
			unsigned char g = unsigned char(double(255) + (percent*(double(pos.g) - double(255))));
			unsigned char b = unsigned char(double(255) + (percent*(double(pos.b) - double(255))));
			//unsigned char a = unsigned char(double(255) + (percent*(double(pos.a) - double(255))));
			buffer[row * k + col] = ((r << 16) | ((g << 8) | b));
		} else {
			unsigned char r = unsigned char(double(255) + (-percent*(double(neg.r) - double(255))));
			unsigned char g = unsigned char(double(255) + (-percent*(double(neg.g) - double(255))));
			unsigned char b = unsigned char(double(255) + (-percent*(double(neg.b) - double(255))));
			//unsigned char a = unsigned char(double(255) + (-percent*(double(neg.a) - double(255))));
			buffer[row * k + col] = ((r << 16) | ((g << 8) | b));
		}
	}
}
void d_drawPixels(int * buffer, int m,int k, const double* vals, bool discrete){ 
	Color pos = Color(100, 167, 211, 255);
	Color neg = Color(255, 184, 113, 255);
	drawPixels_Kernel << <dimGrid(m, k), dimBlock() >> >
		(buffer, k, vals, discrete, neg, pos);
	d_catchErr();
}

__global__ void Sigmoid_Kernal(double *dst, int N) {
	int tid = blockIdx.x;
	if(tid < N)
		dst[tid] = 1/(1+exp(-dst[tid]));
}

__global__ void Tanh_Kernal(double *dst, int N) {
		int tid = blockIdx.x;
		if(tid < N)
			dst[tid] = tanh(dst[tid]);
}

__global__ void ReLU_Kernal(double *dst, int N) {
	int tid = blockIdx.x;
	if(tid < N)
		dst[tid] = max(0.f, dst[tid]);
}

__global__ void LReLU_Kernal(double *dst, int N) {
	int tid = blockIdx.x;
	if(tid < N)
		dst[tid] = max(dst[tid] * LRELU_LEAK, dst[tid]);
}

void d_Activate(d_Matrix *dst, Activation act) {
	switch(act) {
	case Sigmoid:
		Sigmoid_Kernal << < dst->size(), 1 >> > (dst->d_data(), dst->size());
	case Tanh:
		Tanh_Kernal << < dst->size(), 1 >> > (dst->d_data(), dst->size());
		break;
	case ReLU:
		ReLU_Kernal << <dst->size(), 1 >> > (dst->d_data(), dst->size());
		break;
	case LReLU:
		LReLU_Kernal << < dst->size(), 1 >> > (dst->d_data(), dst->size());
		break;
	case Linear: //fall through
	default:
		break;
	}
}

__global__ void backSigmoid_Kernel(double *dst, double *d_W, double *d_dZ, const double *d_A, int m, int n, int k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(col < k && row < m) {
		double sum = 0.0;
		for(int i = 0; i < n; ++i) {
			sum += d_W[row + m * i] * d_dZ[i * k + col];
		}
		dst[row * k + col] = sum * (1 - d_A[row * k + col]);
	}
} /* dst = (d_W.T * d_dZ) (*) d_A^2 */
void d_backSigmoid(d_Matrix *dst, d_Matrix *d_W, d_Matrix *d_dZ, d_Matrix *d_A) {
	int m = d_W->cols(); //reverse for transpose
	int n = d_W->rows(); //reverse for transpose
	int k = d_dZ->cols();
	backSigmoid_Kernel << <dimGrid(m, k), dimBlock() >> > (dst->d_data(), d_W->d_data(), d_dZ->d_data(), d_A->d_data(), m, n, k);
	d_catchErr();
}

__global__ void backTanh_Kernel(double *dst, double *d_W, double *d_dZ, const double *d_A, int m, int n, int k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(col < k && row < m) {
		double sum = 0.0;
		for(int i = 0; i < n; ++i) {
			sum += d_W[row + m * i] * d_dZ[i * k + col];
		}
		dst[row * k + col] = sum * (1 - d_A[row * k + col] * d_A[row * k + col]);
	}
} /* dst = (d_W.T * d_dZ) (*) d_A^2 */
void d_backTanh(d_Matrix *dst, d_Matrix *d_W, d_Matrix *d_dZ, d_Matrix *d_A) {
	int m = d_W->cols(); //reverse for transpose
	int n = d_W->rows(); //reverse for transpose
	int k = d_dZ->cols();
	backTanh_Kernel << <dimGrid(m, k), dimBlock() >> > (dst->d_data(), d_W->d_data(), d_dZ->d_data(), d_A->d_data(), m, n, k);
	d_catchErr();
}

__global__ void backReLU_Kernel(double *dst, double *d_W, double *d_dZ, const double *d_A, int m, int n, int k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(col < k && row < m) {
		double sum = 0.0;
		for(int i = 0; i < n; ++i) {
			sum += d_W[row + m * i] * d_dZ[i * k + col];
		}
		dst[row * k + col] = sum *(d_A[row * k + col] > 0.f ? 1.f : 0.f);
	}
} /* dst = (d_W.T * d_dZ) (*) (d_A > 0 ? 1 : 0) */
void d_backReLU(d_Matrix *dst, d_Matrix *d_W, d_Matrix *d_dZ, d_Matrix *d_A) {
	int m = d_W->cols(); //reverse for transpose
	int n = d_W->rows(); //reverse for transpose
	int k = d_dZ->cols();
	backReLU_Kernel << <dimGrid(m, k), dimBlock() >> > 
		(dst->d_data(), d_W->d_data(), d_dZ->d_data(), d_A->d_data(), m, n, k);
	d_catchErr();
}

__global__ void backLReLU_Kernel(double *dst, double *d_W, double *d_dZ, const double *d_A, int m, int n, int k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(col < k && row < m) {
		double sum = 0.0;
		for(int i = 0; i < n; ++i) {
			sum += d_W[row + m * i] * d_dZ[i * k + col];
		}
		dst[row * k + col] = sum * (d_A[row * k + col] > 0.f ? 1.f : LRELU_LEAK);
	}
} /* dst = (d_W.T * d_dZ) (*) (d_A > 0 ? 1 : 0) */
void d_backLReLU(d_Matrix *dst, d_Matrix *d_W, d_Matrix *d_dZ, d_Matrix *d_A) {
	int m = d_W->cols(); //reverse for transpose
	int n = d_W->rows(); //reverse for transpose
	int k = d_dZ->cols();
	backLReLU_Kernel << <dimGrid(m, k), dimBlock() >> >
		(dst->d_data(), d_W->d_data(), d_dZ->d_data(), d_A->d_data(), m, n, k);
	d_catchErr();
}

__global__ void set_dW_Kernel(double *dst, const double *d_dZ, const double *d_A, double coefficient, int m, int n, int k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	double sum = 0.0;
	if(col < k && row < m) {
		for(int i = 0; i < n; i++) {
			sum += d_dZ[row * n + i] * d_A[col * n + i];
		}
		dst[row * k + col] = sum * coefficient;
	}
} /* dst = coeff * (d_dZ * d_A.T) */
void d_set_dW(d_Matrix* dst, d_Matrix* d_dZ, d_Matrix* d_A, double coefficient) {
	int m = d_dZ->rows();
	int n = d_dZ->cols();
	int k = d_A->rows();
	set_dW_Kernel << <dimGrid(m, k), dimBlock() >> >
		(dst->d_data(), d_dZ->d_data(), d_A->d_data(), coefficient, m, n, k);
	d_catchErr();
}

__global__ void set_dW_Kernel(double *dst, const double *d_dZ, const double *d_A, const double *d_W, double coefficient, double regTerm, int m, int n, int k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	double sum = 0.0;
	if(col < k && row < m) {
		for(int i = 0; i < n; i++) {
			sum += d_dZ[row * n + i] * d_A[col * n + i];
		}
		dst[row * k + col] = coefficient * (sum + (0.5 * regTerm * d_W[row * k + col]));
	}
} /* dst = coeff * (d_dZ * d_A.T) (+) (0.5 * learn * d_W) */
void d_set_dW(d_Matrix* dst, d_Matrix* d_dZ, d_Matrix* d_A, d_Matrix *d_W, double coefficient, double regTerm) {
	int m = d_dZ->rows();
	int n = d_dZ->cols();
	int k = d_A->rows();
	set_dW_Kernel << <dimGrid(m, k), dimBlock() >> >
		(dst->d_data(), d_dZ->d_data(), d_A->d_data(), d_W->d_data(), coefficient, regTerm, m, n, k);
	d_catchErr();
}

__global__ void set_db_Kernel(double *dst, const double *d_dZ, double coefficient, int r, int c) {
	int tid = blockIdx.x;
	if(tid < r) {
		double tempSum = 0.0;
		for(int ind = 0; ind < c; ++ind) {
			tempSum += d_dZ[tid * c + ind];
		}
		dst[tid] = tempSum * coefficient;
	}
} /* dst = coeff * (srcA.SumOfRows) */
void d_set_db(d_Matrix* dst, d_Matrix* d_dZ, double coefficient) {
	set_db_Kernel << <dst->rows(), 1 >> >
		(dst->d_data(), d_dZ->d_data(), coefficient, d_dZ->rows(), d_dZ->cols());
	d_catchErr();
}

#define EPSILON 0.0000000000000001
#define BETA1 0.9
#define BETA2 (1.0 - EPSILON)
__global__ void updateParameterADAM_Kernel(double *dst, int N, const double *d_derivative, double *d_momentum, double *d_momentumSqr, double learn) {
	int tid = blockIdx.x;
	if(tid < N) {
		d_momentum[tid] = BETA1 * d_momentum[tid] + (1 - BETA1) * d_derivative[tid];
		d_momentumSqr[tid] = (BETA2 * d_momentumSqr[tid]) + ((1 - BETA2) * (d_derivative[tid] * d_derivative[tid]));
		dst[tid] -= learn * (d_momentum[tid]/(1-(BETA1*BETA1)) / (sqrt(d_momentumSqr[tid]/(1-(BETA2*BETA2))) + EPSILON));
	}
}
void d_updateParameterADAM(d_Matrix* dst, d_Matrix* d_derivative, d_Matrix* d_momentum, d_Matrix* d_momentumSqr, double learnRate) {
	updateParameterADAM_Kernel << <dst->size(), 1 >> >
		(dst->d_data(), dst->size(), d_derivative->d_data(), d_momentum->d_data(), d_momentumSqr->d_data(), learnRate);
	d_catchErr();
}

__global__ void updateParameter_Kernel(double *dst, int N, const double *d_derivative, double learn) {
	int tid = blockIdx.x;
	if(tid < N) {
		dst[tid] -= learn * d_derivative[tid];
	}
}
void d_updateParameter(d_Matrix* dst, d_Matrix* d_derivative, double learnRate) {
	updateParameter_Kernel << <dst->size(), 1 >> >
		(dst->d_data(), dst->size(), d_derivative->d_data(), learnRate);
	d_catchErr();
}

__global__ void sum_Kernel(double *dst, double *src, int len) {
	int tid = threadIdx.x;
	int step = 2;
	int get = 1;
	while(get < len) {
		if (tid * step + get < len) {
			src[tid*step] += src[tid*step + get];
			__syncthreads();
		}
		step *= 2;
		get *= 2;
	}
	if(tid == 0) {
		if(len % 2 > 0) {
			src[0] += src[len];
		}
		dst[0] = src[0];		
	}
}

__global__ void sumMatrix_Kernel(double *dst, double *src, int m, int k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int len = m * k;
	int step = 2;
	int get = 1;
	int tid = col + row * k;
	while(get < len) {
		if( tid * step + get < len) {
			src[tid*step] += src[tid*step + get];
			__syncthreads();
		}
		step *= 2;
		get *= 2;
		__syncthreads();
	}
	if(tid == 0) {
		if(len % 2 > 0) {
			src[0] += src[len];
		}
		dst[0] = src[0];
	}
}

void d_sum(double *dst, d_Matrix* src) {
	int m = src->size();
	d_Matrix sums = *src;
	sum_Kernel << < 1, m / 2 >> > (dst, src->d_data(), m);
	sums.free();
}


__global__ void square_Kernel(double *dst, double *src, int m, int k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(col < k && row < m) {
		dst[row * k + col] = src[row * k + col] * src[row * k + col];
	}
}

__global__ void square_Kernel(double *dst, double *src) {
	int tid = threadIdx.x;
	dst[tid] = src[tid] * src[tid];
}
void d_square(double *dst, d_Matrix* src) {
	square_Kernel << <1, src->size() >> > (dst, src->d_data());
}

__global__ void finalCost_Kernel(double *dst, double *sumTotal, double regMult, double trainCount) {
	dst[0] = dst[0] + (0.5 * (regMult * (sumTotal[0] / (trainCount*2.0))));
}

__global__ void setZero_Kernel(double *dst) {
	dst[0] = 0.0;
}

void d_calcCost(double *dst, d_Matrix* d_modelErr, vector<d_Matrix>* d_modelWeights, double regMult, double coeff, double trainLableCount) {
	double* d_diff;	
	cudaMalloc((void**)&d_diff, d_modelErr->memSize());
	int m = d_modelErr->size();
	square_Kernel << <1,m >> > (d_diff, d_modelErr->d_data()); d_catchErr();
	sum_Kernel << <1, m/2 >> > (dst, d_diff, m); d_catchErr();
	mult_elem_Kernel << <1, 1 >> > (dst, dst, coeff); d_catchErr();
	return;
	// Add Regularization
	double* d_sqrSumTotal;
	cudaMalloc((void**)&d_sqrSumTotal, sizeof(double));
	setZero_Kernel << <1, 1 >> > (d_sqrSumTotal); d_catchErr();
	for(int i = 0; i < (int)d_modelWeights->size()-1; ++i) {
		int m = d_modelWeights->at(i).rows();
		int k = d_modelWeights->at(i).cols();
		double* d_squared;
		double* d_sqrSum;
		d_check(cudaMalloc((void**)&d_sqrSum, sizeof(double)));
		d_check(cudaMalloc((void**)&d_squared, d_modelWeights->at(i).memSize()));		
		square_Kernel << < dimGrid(m, k), dimBlock() >> > (d_squared, d_modelWeights->at(i).d_data(), m, k); d_catchErr();
		sumMatrix_Kernel << <dimGrid(m, k), dimBlock() >> > (d_sqrSum, d_squared, m, k); d_catchErr();
		add_Kernel << <1, 1 >> > (d_sqrSumTotal, d_sqrSumTotal, d_sqrSum); d_catchErr();
		d_check(cudaFree(d_sqrSum));
		d_check(cudaFree(d_squared));
	}
	finalCost_Kernel << <1, 1 >> > (dst, d_sqrSumTotal, regMult, trainLableCount); d_catchErr();
	cudaFree(d_sqrSumTotal);
	cudaFree(d_diff); 
	d_catchErr();
}