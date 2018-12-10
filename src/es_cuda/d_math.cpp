#include "d_math.h"
#include <stdio.h>


dim3 dimGrid(int m, int k) {
	return dim3((k + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);
}

dim3 dimBlock() {
	return dim3(BLOCK_SIZE, BLOCK_SIZE);
}

__global__ void addKernel(float *c, const float *a, const float *b) {
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
} /* dst = srcA (+) srcB */
void d_add(d_MatrixXf *dst, d_MatrixXf *srcA, d_MatrixXf *srcB) {
	addKernel << <1, dst->size() >> > (dst->d_data(), srcA->d_data(), srcB->d_data());
}


__global__ void subtractKernel(float *c, const float *a, const float *b) {
	int i = threadIdx.x;
	c[i] = a[i] - b[i];
} /* dst = srcA (-) srcB */
void d_subtract(d_MatrixXf *dst, d_MatrixXf *srcA, d_MatrixXf *srcB) {
	subtractKernel << <1, dst->size() >> > (dst->d_data(), srcA->d_data(), srcB->d_data());
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
} /* dst = srcA * srcB */
void d_matrixMult(d_MatrixXf* dst, d_MatrixXf* srcA, d_MatrixXf* srcB){
	int m = srcA->rows();
	int n = srcA->cols();
	int k = srcB->cols();
	MatrixMultKernel << <dimGrid(m, k), dimBlock() >> >
		(dst->d_data(), srcA->d_data(), srcB->d_data(), m, n, k);
}

__global__ void MatrixMult_lhsT_Kernel(float *dst, float *srcA, float *srcB, int m, int n, int k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(col < k && row < m) {
		float tempSum = 0.f;
		for(int ind = 0; ind < n; ++ind) {
			tempSum += srcA[row * n + ind] * srcB[col * n + ind];
		}
		dst[col * m + row] = tempSum;
	}
} /*dst = srcA.T * srcB */
void d_matrixMult_lhsT(d_MatrixXf* dst, d_MatrixXf* srcA, d_MatrixXf* srcB) {
	int m = srcA->cols(); //reverse for transpose
	int n = srcA->rows(); //reverse for transpose
	int k = srcB->cols();
	MatrixMult_lhsT_Kernel << <dimGrid(m, k), dimBlock() >> >
		(dst->d_data(), srcA->d_data(), srcB->d_data(), m, n, k);
}

__global__ void MatrixMult_rhsT_Kernel(float *dst, const float *srcA, const float *srcB, int m, int n, int k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(col < k && row < m) {
		float tempSum = 0.f;
		for(int ind = 0; ind < n; ++ind) {
			tempSum += srcA[row + m * ind] * srcB[col + k * ind];
		}
		dst[col * m + row] = tempSum;
	}
} /* dst = srcA * srcB.T */
void d_matrixMult_rhsT(d_MatrixXf* dst, d_MatrixXf* srcA, d_MatrixXf* srcB) {
	int m = srcA->rows();
	int n = srcA->cols();
	int k = srcB->rows(); //reverse for transpose
	MatrixMult_rhsT_Kernel << <dimGrid(m, k), dimBlock() >> >
		(dst->d_data(), srcA->d_data(), srcB->d_data(), m, n, k);
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
} /* dst = d_W * d_last + d_bias */
void d_forwardLayer(d_MatrixXf *dst, d_MatrixXf *d_W, d_MatrixXf *d_last, d_MatrixXf *d_bias) {
	int m = d_W->rows();
	int n = d_W->cols();
	int k = d_last->cols();
	ForwardLayerKernel << <dimGrid(m, k), dimBlock() >> >
		(dst->d_data(), d_W->d_data(), d_last->d_data(), d_bias->d_data(), m, n, k);
}


__global__ void DrawPixelsKernel(int *buffer, int m, const float* vals, bool discrete, Color neg, Color pos) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	float percent = vals[col * m + row];
	if(discrete) {
		if (percent > 0.f) {
			buffer[col * m + row] = ((pos.r << 16) | ((pos.g << 8) | pos.b));
		} else {
			buffer[col * m + row] = ((neg.r << 16) | ((neg.g << 8) | neg.b));
		}
	} else {
		if(percent > 0.f) {
			unsigned char r = unsigned char(float(255) + (percent*(float(pos.r) - float(255))));
			unsigned char g = unsigned char(float(255) + (percent*(float(pos.g) - float(255))));
			unsigned char b = unsigned char(float(255) + (percent*(float(pos.b) - float(255))));
			//unsigned char a = unsigned char(float(255) + (percent*(float(pos.a) - float(255))));
			buffer[col * m + row] = ((r << 16) | ((g << 8) | b));
		} else {
			unsigned char r = unsigned char(float(255) + (-percent*(float(neg.r) - float(255))));
			unsigned char g = unsigned char(float(255) + (-percent*(float(neg.g) - float(255))));
			unsigned char b = unsigned char(float(255) + (-percent*(float(neg.b) - float(255))));
			//unsigned char a = unsigned char(float(255) + (-percent*(float(neg.a) - float(255))));
			buffer[col * m + row] = ((r << 16) | ((g << 8) | b));
		}
	}
}
void d_drawPixels(int * buffer, int m,int k, const float* vals, bool discrete){ 
	Color pos = Color(100, 167, 211, 255);
	Color neg = Color(255, 184, 113, 255);
	DrawPixelsKernel << <dimGrid(m, k), dimBlock() >> >
		(buffer, m, vals, discrete, neg, pos);
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
		dst[tid] = max(dst[tid] * LRELU_LEAK, dst[tid]);
}

void d_Activate(d_MatrixXf *dst, Activation act) {
	switch(act) {
	case Sigmoid:
		SigmoidKernal << < dst->size(), 1 >> > (dst->d_data(), dst->size());
	case Tanh:
		TanhKernal << < dst->size(), 1 >> > (dst->d_data(), dst->size());
		break;
	case ReLU:
		ReLUKernal << <dst->size(), 1 >> > (dst->d_data(), dst->size());
		break;
	case LReLU:
		LReLUKernal << < dst->size(), 1 >> > (dst->d_data(), dst->size());
		break;
	case Linear: //fall through
	default:
		break;
	}
}

__global__ void BackSigmoidKernel(float *dst, float *d_W, float *d_dZ, const float *d_A, int m, int n, int k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(col < k && row < m) {
		float tempSum = 0.f;
		for(int ind = 0; ind < n; ++ind) {
			tempSum += d_W[row * n + ind] * d_dZ[col * n + ind];
		}
		dst[col * m + row] = tempSum * (1 - d_A[col * m + row]);
	}
} /* dst = (d_W.T * d_dZ) (*) d_A^2 */
void d_BackSigmoid(d_MatrixXf *dst, d_MatrixXf *d_W, d_MatrixXf *d_dZ, d_MatrixXf *d_A) {
	int m = d_W->cols(); //reverse for transpose
	int n = d_W->rows(); //reverse for transpose
	int k = d_dZ->cols();
	BackSigmoidKernel << <dimGrid(m, k), dimBlock() >> > (dst->d_data(), d_W->d_data(), d_dZ->d_data(), d_A->d_data(), m, n, k);
}

__global__ void BackTanhKernel(float *dst, float *d_W, float *d_dZ, const float *d_A, int m, int n, int k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(col < k && row < m) {
		float tempSum = 0.f;
		for(int ind = 0; ind < n; ++ind) {
			tempSum += d_W[row * n + ind] * d_dZ[col * n + ind];
		}
		dst[col * m + row] = tempSum * (1 - d_A[col * m + row] * d_A[col * m + row]);
	}
} /* dst = (d_W.T * d_dZ) (*) d_A^2 */
void d_BackTanh(d_MatrixXf *dst, d_MatrixXf *d_W, d_MatrixXf *d_dZ, d_MatrixXf *d_A) {
	int m = d_W->cols(); //reverse for transpose
	int n = d_W->rows(); //reverse for transpose
	int k = d_dZ->cols();
	BackTanhKernel << <dimGrid(m, k), dimBlock() >> > (dst->d_data(), d_W->d_data(), d_dZ->d_data(), d_A->d_data(), m, n, k);
}

__global__ void BackReLUKernel(float *dst, float *d_W, float *d_dZ, const float *d_A, int m, int n, int k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(col < k && row < m) {
		float tempSum = 0.f;
		for(int ind = 0; ind < n; ++ind) {
			tempSum += d_W[row * n + ind] * d_dZ[col * n + ind];
		}
		dst[col * m + row] = tempSum *(d_A[col * m + row] > 0.f ? 1.f : 0.f);
	}
} /* dst = (d_W.T * d_dZ) (*) (d_A > 0 ? 1 : 0) */
void d_BackReLU(d_MatrixXf *dst, d_MatrixXf *d_W, d_MatrixXf *d_dZ, d_MatrixXf *d_A) {
	int m = d_W->cols(); //reverse for transpose
	int n = d_W->rows(); //reverse for transpose
	int k = d_dZ->cols();
	BackReLUKernel << <dimGrid(m, k), dimBlock() >> > 
		(dst->d_data(), d_W->d_data(), d_dZ->d_data(), d_A->d_data(), m, n, k);
}

__global__ void BackLReLUKernel(float *dst, float *d_W, float *d_dZ, const float *d_A, int m, int n, int k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(col < k && row < m) {
		float tempSum = 0.f;
		for(int ind = 0; ind < n; ++ind) {
			tempSum += d_W[row * n + ind] * d_dZ[col * n + ind];
		}
		dst[col * m + row] = tempSum *(d_A[col * m + row] > 0.f ? 1.f : LRELU_LEAK);
	}
} /* dst = (d_W.T * d_dZ) (*) (d_A > 0 ? 1 : 0) */
void d_BackLReLU(d_MatrixXf *dst, d_MatrixXf *d_W, d_MatrixXf *d_dZ, d_MatrixXf *d_A) {
	int m = d_W->cols(); //reverse for transpose
	int n = d_W->rows(); //reverse for transpose
	int k = d_dZ->cols();
	BackLReLUKernel << <dimGrid(m, k), dimBlock() >> >
		(dst->d_data(), d_W->d_data(), d_dZ->d_data(), d_A->d_data(), m, n, k);
}

__global__ void Set_dW_Kernel(float *dst, const float *d_dZ, const float *d_A, float coefficient, int m, int n, int k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(col < k && row < m) {
		float tempSum = 0.f;
		for(int ind = 0; ind < n; ++ind) {
			tempSum += d_dZ[row + m * ind] * d_A[col + k * ind];
		}
		dst[col * m + row] = tempSum * coefficient;
	}
} /* dst = coeff * (d_dZ * d_A.T) */
void d_Set_dW(d_MatrixXf* dst, d_MatrixXf* d_dZ, d_MatrixXf* d_A, float coefficient) {
	int m = d_dZ->rows();
	int n = d_dZ->cols();
	int k = d_A->rows();
	Set_dW_Kernel << <dimGrid(m, k), dimBlock() >> >
		(dst->d_data(), d_dZ->d_data(), d_A->d_data(), coefficient, m, n, k);
}
 
__global__ void Set_dW_Kernel(float *dst, const float *d_dZ, const float *d_A, const float *d_W, float coefficient, float regTerm, int m, int n, int k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(col < k && row < m) {
		float tempSum = 0.f;
		for(int ind = 0; ind < n; ++ind) {
			tempSum += d_dZ[row + m * ind] * d_A[col + k * ind];
		}
		dst[col * m + row] = coefficient * (tempSum + (0.5 * regTerm * d_W[col * m + row]));
	}
} /* dst = coeff * (d_dZ * d_A.T) (+) (0.5 * learn * d_W) */
void d_Set_dW(d_MatrixXf* dst, d_MatrixXf* d_dZ, d_MatrixXf* d_A, d_MatrixXf *d_W, float coefficient, float regTerm) {
	int m = d_dZ->rows();
	int n = d_dZ->cols();
	int k = d_A->rows();
	Set_dW_Kernel << <dimGrid(m, k), dimBlock() >> >
		(dst->d_data(), d_dZ->d_data(), d_A->d_data(), d_W->d_data(), coefficient, regTerm, m, n, k);
}

__global__ void Set_db_Kernel(float *dst, const float *d_dZ, float coefficient, int r, int c) {
	int tid = blockIdx.x;
	if(tid < r*c){ 
		float tempSum = 0.f;
		for(int ind = 0; ind < c; ++ind) {
			tempSum += d_dZ[r * ind];
		}
	dst[tid] = tempSum * coefficient;
	}
} /* dst = coeff * (srcA.SumOfRows) */
void d_Set_db(d_MatrixXf* dst, d_MatrixXf* d_dZ, float coefficient) {
	Set_db_Kernel << <d_dZ->rows(), 1 >> >
		(dst->d_data(), d_dZ->d_data(), coefficient, d_dZ->rows(), d_dZ->cols());
}