#include "d_math.h"
#include <stdio.h>
dim3 dimGrid(int m, int k){
	return dim3((k + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);
}
dim3 dimBlock(){
	return dim3(BLOCK_SIZE, BLOCK_SIZE);
}
__global__ void add_Kernel(float *c, const float *a, const float *b){
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
} /* dst = srcA (+) srcB */
void d_add(d_Matrix *dst, d_Matrix *srcA, d_Matrix *srcB){
	add_Kernel << <1, dst->size() >> > (dst->d_data(), srcA->d_data(), srcB->d_data());
	d_catchErr();
}
__global__ void subtract_Kernel(float *c, const float *a, const float *b){
	int i = threadIdx.x;
	c[i] = a[i] - b[i];
} /* dst = srcA (-) srcB */
void d_subtract(d_Matrix *dst, d_Matrix *srcA, d_Matrix *srcB){
	subtract_Kernel << <1, dst->size() >> > (dst->d_data(), srcA->d_data(), srcB->d_data());
	d_catchErr();
}
__global__ void mult_elem_Kernel(float *c, float *a, float b){
	int i = threadIdx.x;
	c[i] = a[i] * b;
}
__global__ void mult_Kernel(float *dst, float *srcA, float *srcB, int m, int n, int k){
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	float sum = 0.0;
	if(col < k && row < m){
		for(int i = 0; i < n; i++){
			sum += srcA[row * n + i] * srcB[i * k + col];
		}
		dst[row * k + col] = sum;
	}
}
void d_mult(d_Matrix* dst, d_Matrix* srcA, d_Matrix* srcB){
	int m = srcA->rows();
	int n = srcA->cols();
	int k = srcB->cols();
	mult_Kernel << <dimGrid(m, k), dimBlock() >> >
		(dst->d_data(), srcA->d_data(), srcB->d_data(), m, n, k);
	d_catchErr();
}
__global__ void mult_lhsT_Kernel(float *dst, float *srcA, float *srcB, int m, int n, int k){
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(col < k && row < m){
		float sum = 0.f;
		for(int i = 0; i < n; ++i){
			sum += srcA[row + m * i] * srcB[i * k + col];
		}
		dst[row * k + col] = sum;
	}
} /*dst = srcA.T * srcB */
void d_mult_lhsT(d_Matrix* dst, d_Matrix* srcA, d_Matrix* srcB){
	int m = srcA->cols(); //reverse for transpose
	int n = srcA->rows(); //reverse for transpose
	int k = srcB->cols();
	mult_lhsT_Kernel << <dimGrid(m, k), dimBlock() >> >
		(dst->d_data(), srcA->d_data(), srcB->d_data(), m, n, k);
	d_catchErr();
}
__global__ void mult_rhsT_Kernel(float *dst, const float *srcA, const float *srcB, int m, int n, int k){
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	float sum = 0.0;
	if(col < k && row < m){
		for(int i = 0; i < n; i++){
			sum += srcA[row * n + i] * srcB[col * n + i];
		}
		dst[row * k + col] = sum;
	}
} /* dst = srcA * srcB.T */
void d_mult_rhsT(d_Matrix* dst, d_Matrix* srcA, d_Matrix* srcB){
	int m = srcA->rows();
	int n = srcA->cols();
	int k = srcB->rows(); //reverse for transpose
	mult_rhsT_Kernel << <dimGrid(m, k), dimBlock() >> >
		(dst->d_data(), srcA->d_data(), srcB->d_data(), m, n, k);
	d_catchErr();
}
__global__ void forwardLayer_Kernel(float *dst, const float *d_W, const float *d_last, const float * d_bias, int m, int n, int k){
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	float sum = 0.f;
	if(col < k && row < m){
		for(int i = 0; i < n; i++){
			sum += d_W[row * n + i] * d_last[i * k + col];
		}
		dst[row * k + col] = sum + d_bias[row];
	}
} /* dst = d_W * d_last + d_bias */
void d_forwardLayer(d_Matrix *dst, d_Matrix *d_W, d_Matrix *d_last, d_Matrix *d_bias){
	int m = d_W->rows();
	int n = d_W->cols();
	int k = d_last->cols();
	forwardLayer_Kernel << <dimGrid(m, k), dimBlock() >> >
		(dst->d_data(), d_W->d_data(), d_last->d_data(), d_bias->d_data(), m, n, k);
	d_catchErr();
}
__global__ void drawPixels_Kernel(int *buffer, int k, const float* vals, bool discrete, Color neg, Color pos){
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	float percent = vals[row * k + col];
	if(discrete){
		if(percent > 0.f){
			buffer[row * k + col] = ((pos.r << 16) | ((pos.g << 8) | pos.b));
		} else{
			buffer[row * k + col] = ((neg.r << 16) | ((neg.g << 8) | neg.b));
		}
	} else{
		if(percent > 0.){
			unsigned char r = unsigned char(float(255) + (percent*(float(pos.r) - float(255))));
			unsigned char g = unsigned char(float(255) + (percent*(float(pos.g) - float(255))));
			unsigned char b = unsigned char(float(255) + (percent*(float(pos.b) - float(255))));
			//unsigned char a = unsigned char(float(255) + (percent*(float(pos.a) - float(255))));
			buffer[row * k + col] = ((r << 16) | ((g << 8) | b));
		} else{
			unsigned char r = unsigned char(float(255) + (-percent*(float(neg.r) - float(255))));
			unsigned char g = unsigned char(float(255) + (-percent*(float(neg.g) - float(255))));
			unsigned char b = unsigned char(float(255) + (-percent*(float(neg.b) - float(255))));
			//unsigned char a = unsigned char(float(255) + (-percent*(float(neg.a) - float(255))));
			buffer[row * k + col] = ((r << 16) | ((g << 8) | b));
		}
	}
}
void d_drawPixels(int * buffer, int m, int k, const float* vals, bool discrete){
	Color pos = Color(100, 167, 211, 255);
	Color neg = Color(255, 184, 113, 255);
	drawPixels_Kernel << <dimGrid(m, k), dimBlock() >> >
		(buffer, k, vals, discrete, neg, pos);
	d_catchErr();
}
__global__ void Sigmoid_Kernal(float *dst, int N){
	int tid = blockIdx.x;
	if(tid < N)
		dst[tid] = 1 / (1 + exp(-dst[tid]));
}
__global__ void Tanh_Kernal(float *dst, int N){
	int tid = blockIdx.x;
	if(tid < N)
		dst[tid] = tanh(dst[tid]);
}
__global__ void ReLU_Kernal(float *dst, int N){
	int tid = blockIdx.x;
	if(tid < N)
		dst[tid] = max(0.f, dst[tid]);
}
__global__ void LReLU_Kernal(float *dst, int N){
	int tid = blockIdx.x;
	if(tid < N)
		dst[tid] = max(dst[tid] * LRELU_LEAK, dst[tid]);
}
void d_activate(d_Matrix *dst, Activation act){
	switch(act){
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
__global__ void backSigmoid_Kernel(float *dst, float *d_W, float *d_dZ, const float *d_A, int m, int n, int k){
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(col < k && row < m){
		float sum = 0.f;
		for(int i = 0; i < n; ++i){
			sum += d_W[row + m * i] * d_dZ[i * k + col];
		}
		dst[row * k + col] = sum * (1 - d_A[row * k + col]);
	}
} /* dst = (d_W.T * d_dZ) (*) d_A^2 */
void d_backSigmoid(d_Matrix *dst, d_Matrix *d_W, d_Matrix *d_dZ, d_Matrix *d_A){
	int m = d_W->cols(); //reverse for transpose
	int n = d_W->rows(); //reverse for transpose
	int k = d_dZ->cols();
	backSigmoid_Kernel << <dimGrid(m, k), dimBlock() >> > (dst->d_data(), d_W->d_data(), d_dZ->d_data(), d_A->d_data(), m, n, k);
	d_catchErr();
}
__global__ void backTanh_Kernel(float *dst, float *d_W, float *d_dZ, const float *d_A, int m, int n, int k){
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(col < k && row < m){
		float sum = 0.f;
		for(int i = 0; i < n; ++i){
			sum += d_W[row + m * i] * d_dZ[i * k + col];
		}
		dst[row * k + col] = sum * (1 - d_A[row * k + col] * d_A[row * k + col]);
	}
} /* dst = (d_W.T * d_dZ) (*) d_A^2 */
void d_backTanh(d_Matrix *dst, d_Matrix *d_W, d_Matrix *d_dZ, d_Matrix *d_A){
	int m = d_W->cols(); //reverse for transpose
	int n = d_W->rows(); //reverse for transpose
	int k = d_dZ->cols();
	backTanh_Kernel << <dimGrid(m, k), dimBlock() >> > (dst->d_data(), d_W->d_data(), d_dZ->d_data(), d_A->d_data(), m, n, k);
	d_catchErr();
}
__global__ void backReLU_Kernel(float *dst, float *d_W, float *d_dZ, const float *d_A, int m, int n, int k){
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(col < k && row < m){
		float sum = 0.f;
		for(int i = 0; i < n; ++i){
			sum += d_W[row + m * i] * d_dZ[i * k + col];
		}
		dst[row * k + col] = sum *(d_A[row * k + col] > 0.f ? 1.f : 0.f);
	}
} /* dst = (d_W.T * d_dZ) (*) (d_A > 0 ? 1 : 0) */
void d_backReLU(d_Matrix *dst, d_Matrix *d_W, d_Matrix *d_dZ, d_Matrix *d_A){
	int m = d_W->cols(); //reverse for transpose
	int n = d_W->rows(); //reverse for transpose
	int k = d_dZ->cols();
	backReLU_Kernel << <dimGrid(m, k), dimBlock() >> >
		(dst->d_data(), d_W->d_data(), d_dZ->d_data(), d_A->d_data(), m, n, k);
	d_catchErr();
}
__global__ void backLReLU_Kernel(float *dst, float *d_W, float *d_dZ, const float *d_A, int m, int n, int k){
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(col < k && row < m){
		float sum = 0.f;
		for(int i = 0; i < n; ++i){
			sum += d_W[row + m * i] * d_dZ[i * k + col];
		}
		dst[row * k + col] = sum * (d_A[row * k + col] > 0.f ? 1.f : LRELU_LEAK);
	}
} /* dst = (d_W.T * d_dZ) (*) (d_A > 0 ? 1 : 0) */
void d_backLReLU(d_Matrix *dst, d_Matrix *d_W, d_Matrix *d_dZ, d_Matrix *d_A){
	int m = d_W->cols(); //reverse for transpose
	int n = d_W->rows(); //reverse for transpose
	int k = d_dZ->cols();
	backLReLU_Kernel << <dimGrid(m, k), dimBlock() >> >
		(dst->d_data(), d_W->d_data(), d_dZ->d_data(), d_A->d_data(), m, n, k);
	d_catchErr();
}
__global__ void set_dW_Kernel(float *dst, const float *d_dZ, const float *d_A, float coefficient, int m, int n, int k){
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	float sum = 0.f;
	if(col < k && row < m){
		for(int i = 0; i < n; i++){
			sum += d_dZ[row * n + i] * d_A[col * n + i];
		}
		dst[row * k + col] = sum * coefficient;
	}
} /* dst = coeff * (d_dZ * d_A.T) */
void d_set_dW(d_Matrix* dst, d_Matrix* d_dZ, d_Matrix* d_A, float coefficient){
	int m = d_dZ->rows();
	int n = d_dZ->cols();
	int k = d_A->rows();
	set_dW_Kernel << <dimGrid(m, k), dimBlock() >> >
		(dst->d_data(), d_dZ->d_data(), d_A->d_data(), coefficient, m, n, k);
	d_catchErr();
}
__global__ void set_dW_Kernel(float *dst, const float *d_dZ, const float *d_A, const float *d_W, float coefficient, float regTerm, int m, int n, int k){
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	float sum = 0.f;
	if(col < k && row < m){
		for(int i = 0; i < n; i++){
			sum += d_dZ[row * n + i] * d_A[col * n + i];
		}
		dst[row * k + col] = coefficient * (sum + (0.5 * regTerm * d_W[row * k + col]));
	}
} /* dst = coeff * (d_dZ * d_A.T) (+) (0.5 * learn * d_W) */
void d_set_dW(d_Matrix* dst, d_Matrix* d_dZ, d_Matrix* d_A, d_Matrix *d_W, float coefficient, float regTerm){
	int m = d_dZ->rows();
	int n = d_dZ->cols();
	int k = d_A->rows();
	set_dW_Kernel << <dimGrid(m, k), dimBlock() >> >
		(dst->d_data(), d_dZ->d_data(), d_A->d_data(), d_W->d_data(), coefficient, regTerm, m, n, k);
	d_catchErr();
}
__global__ void set_db_Kernel(float *dst, const float *d_dZ, float coefficient, int r, int c){
	int tid = blockIdx.x;
	if(tid < r){
		float sum = 0.f;
		for(int ind = 0; ind < c; ++ind){
			sum += d_dZ[tid * c + ind];
		}
		dst[tid] = sum * coefficient;
	}
} /* dst = coeff * (srcA.SumOfRows) */
void d_set_db(d_Matrix* dst, d_Matrix* d_dZ, float coefficient){
	set_db_Kernel << <dst->rows(), 1 >> >
		(dst->d_data(), d_dZ->d_data(), coefficient, d_dZ->rows(), d_dZ->cols());
	d_catchErr();
}
#define BETA1 0.9f
#define BETA2 (1.f - FLT_EPSILON)
__global__ void updateParameterADAM_Kernel(float *dst, int N, const float *d_derivative, float *d_momentum, float *d_momentumSqr, float learn){
	int tid = blockIdx.x;
	if(tid < N){
		d_momentum[tid] = BETA1 * d_momentum[tid] + (1 - BETA1) * d_derivative[tid];
		d_momentumSqr[tid] = (BETA2 * d_momentumSqr[tid]) + ((1 - BETA2) * (d_derivative[tid] * d_derivative[tid]));
		dst[tid] -= learn * (d_momentum[tid] / (1 - (BETA1*BETA1)) / (sqrt(d_momentumSqr[tid] / (1 - (BETA2*BETA2))) + FLT_EPSILON));
	}
}
void d_updateParameterADAM(d_Matrix* dst, d_Matrix* d_derivative, d_Matrix* d_momentum, d_Matrix* d_momentumSqr, float learnRate){
	updateParameterADAM_Kernel << <dst->size(), 1 >> >
		(dst->d_data(), dst->size(), d_derivative->d_data(), d_momentum->d_data(), d_momentumSqr->d_data(), learnRate);
	d_catchErr();
}
__global__ void updateParameter_Kernel(float *dst, int N, const float *d_derivative, float learn){
	int tid = blockIdx.x;
	if(tid < N){
		dst[tid] -= learn * d_derivative[tid];
	}
}
void d_updateParameter(d_Matrix* dst, d_Matrix* d_derivative, float learnRate){
	updateParameter_Kernel << <dst->size(), 1 >> >
		(dst->d_data(), dst->size(), d_derivative->d_data(), learnRate);
	d_catchErr();
}
__global__ void sum_Kernel(float *dst, float *src, int len){
	int tid = threadIdx.x;
	int step = 2;
	int get = 1;
	while(get < len){
		if(tid * step + get < len){
			src[tid*step] += src[tid*step + get];
			__syncthreads();
		}
		step *= 2;
		get *= 2;
	}
	if(tid == 0){
		if(len % 2 > 0){
			src[0] += src[len];
		}
		dst[0] = src[0];
	}
}
__global__ void sumMatrix_Kernel(float *dst, float *src, int m, int k){
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int len = m * k;
	int step = 2;
	int get = 1;
	int tid = col + row * k;
	while(get < len){
		if(tid * step + get < len){
			src[tid*step] += src[tid*step + get];
			__syncthreads();
		}
		step *= 2;
		get *= 2;
		__syncthreads();
	}
	if(tid == 0){
		if(len % 2 > 0){
			src[0] += src[len];
		}
		dst[0] = src[0];
	}
}
void d_sum(float *dst, d_Matrix* src){
	int m = src->size();
	d_Matrix sums = *src;
	sum_Kernel << < 1, m / 2 >> > (dst, src->d_data(), m);
	sums.free();
}
__global__ void square_Kernel(float *dst, float *src, int m, int k){
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(col < k && row < m){
		dst[row * k + col] = src[row * k + col] * src[row * k + col];
	}
}
__global__ void square_Kernel(float *dst, float *src){
	int tid = threadIdx.x;
	dst[tid] = src[tid] * src[tid];
}
void d_square(float *dst, d_Matrix* src){
	square_Kernel << <1, src->size() >> > (dst, src->d_data());
}
__global__ void finalCost_Kernel(float *dst, float *sumTotal, float regMult, float trainCount){
	dst[0] = dst[0] + (0.5f * (regMult * (sumTotal[0] / (trainCount*2.0f))));
}
__global__ void setZero_Kernel(float *dst){
	dst[0] = 0.0f;
}
void d_calcCost(float *dst, d_Matrix* d_modelErr, vector<d_Matrix>* d_modelWeights, float regMult, float coeff, float trainLableCount){
	float* d_diff;
	cudaMalloc((void**)&d_diff, d_modelErr->memSize());
	int m = d_modelErr->size();
	square_Kernel << <1, m >> > (d_diff, d_modelErr->d_data()); d_catchErr();
	sum_Kernel << <1, m / 2 >> > (dst, d_diff, m); d_catchErr();
	mult_elem_Kernel << <1, 1 >> > (dst, dst, coeff); d_catchErr();
	return;
	// Add Regularization
	float* d_sqrSumTotal;
	cudaMalloc((void**)&d_sqrSumTotal, sizeof(float));
	setZero_Kernel << <1, 1 >> > (d_sqrSumTotal); d_catchErr();
	for(int i = 0; i < (int)d_modelWeights->size() - 1; ++i){
		int m = d_modelWeights->at(i).rows();
		int k = d_modelWeights->at(i).cols();
		float* d_squared;
		float* d_sqrSum;
		d_check(cudaMalloc((void**)&d_sqrSum, sizeof(float)));
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