#include "d_math.h"
#include <stdio.h>


dim3 dimGrid(int m, int k) {
	return dim3((k + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);
}

dim3 dimBlock() {
	return dim3(BLOCK_SIZE, BLOCK_SIZE);
}

__global__ void addKernel(double *c, const double *a, const double *b) {
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
} /* dst = srcA (+) srcB */
void d_add(d_Matrix *dst, d_Matrix *srcA, d_Matrix *srcB) {
	addKernel << <1, dst->size() >> > (dst->d_data(), srcA->d_data(), srcB->d_data());
}


__global__ void subtractKernel(double *c, const double *a, const double *b) {
	int i = threadIdx.x;
	c[i] = a[i] - b[i];
} /* dst = srcA (-) srcB */
void d_subtract(d_Matrix *dst, d_Matrix *srcA, d_Matrix *srcB) {
	subtractKernel << <1, dst->size() >> > (dst->d_data(), srcA->d_data(), srcB->d_data());
}


__global__ void MatrixMultKernel(double *dst,const double *srcA,const double *srcB, int m, int n, int k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(col < k && row < m) {
		double tempSum = 0.0;
		for(int ind = 0; ind < n; ++ind) {
			tempSum += srcA[row + m * ind] * srcB[col * n + ind];
		}
		dst[col * m + row] = tempSum;
	}
} /* dst = srcA * srcB */
void d_matrixMult(d_Matrix* dst, d_Matrix* srcA, d_Matrix* srcB){
	int m = srcA->rows();
	int n = srcA->cols();
	int k = srcB->cols();
	MatrixMultKernel << <dimGrid(m, k), dimBlock() >> >
		(dst->d_data(), srcA->d_data(), srcB->d_data(), m, n, k);
}


__global__ void MatrixMult_lhsT_Kernel(double *dst, double *srcA, double *srcB, int m, int n, int k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(col < k && row < m) {
		double tempSum = 0.0;
		for(int ind = 0; ind < n; ++ind) {
			tempSum += srcA[row * n + ind] * srcB[col * n + ind];
		}
		dst[col * m + row] = tempSum;
	}
} /*dst = srcA.T * srcB */
void d_matrixMult_lhsT(d_Matrix* dst, d_Matrix* srcA, d_Matrix* srcB) {
	int m = srcA->cols(); //reverse for transpose
	int n = srcA->rows(); //reverse for transpose
	int k = srcB->cols();
	MatrixMult_lhsT_Kernel << <dimGrid(m, k), dimBlock() >> >
		(dst->d_data(), srcA->d_data(), srcB->d_data(), m, n, k);
}

__global__ void MatrixMult_rhsT_Kernel(double *dst, const double *srcA, const double *srcB, int m, int n, int k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(col < k && row < m) {
		double tempSum = 0.0;
		for(int ind = 0; ind < n; ++ind) {
			tempSum += srcA[row + m * ind] * srcB[col + k * ind];
		}
		dst[col * m + row] = tempSum;
	}
} /* dst = srcA * srcB.T */
void d_matrixMult_rhsT(d_Matrix* dst, d_Matrix* srcA, d_Matrix* srcB) {
	int m = srcA->rows();
	int n = srcA->cols();
	int k = srcB->rows(); //reverse for transpose
	MatrixMult_rhsT_Kernel << <dimGrid(m, k), dimBlock() >> >
		(dst->d_data(), srcA->d_data(), srcB->d_data(), m, n, k);
}

__global__ void ForwardLayerKernel(double *dst,const double *d_W, const double *d_last, const double * d_bias, int m, int n, int k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(col < k && row < m) {
		double tempSum = 0.0;
		for(int ind = 0; ind < n; ++ind) {
			tempSum += d_W[row + m * ind] * d_last[col * n + ind];
		}
		dst[col * m + row] = tempSum + d_bias[row];
	}
} /* dst = d_W * d_last + d_bias */
void d_forwardLayer(d_Matrix *dst, d_Matrix *d_W, d_Matrix *d_last, d_Matrix *d_bias) {
	int m = d_W->rows();
	int n = d_W->cols();
	int k = d_last->cols();
	ForwardLayerKernel << <dimGrid(m, k), dimBlock() >> >
		(dst->d_data(), d_W->d_data(), d_last->d_data(), d_bias->d_data(), m, n, k);
}


__global__ void DrawPixelsKernel(int *buffer, int m, const double* vals, bool discrete, Color neg, Color pos) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	double percent = vals[col * m + row];
	if(discrete) {
		if (percent > 0.0) {
			buffer[col * m + row] = ((pos.r << 16) | ((pos.g << 8) | pos.b));
		} else {
			buffer[col * m + row] = ((neg.r << 16) | ((neg.g << 8) | neg.b));
		}
	} else {
		if(percent > 0.) {
			unsigned char r = unsigned char(double(255) + (percent*(double(pos.r) - double(255))));
			unsigned char g = unsigned char(double(255) + (percent*(double(pos.g) - double(255))));
			unsigned char b = unsigned char(double(255) + (percent*(double(pos.b) - double(255))));
			//unsigned char a = unsigned char(double(255) + (percent*(double(pos.a) - double(255))));
			buffer[col * m + row] = ((r << 16) | ((g << 8) | b));
		} else {
			unsigned char r = unsigned char(double(255) + (-percent*(double(neg.r) - double(255))));
			unsigned char g = unsigned char(double(255) + (-percent*(double(neg.g) - double(255))));
			unsigned char b = unsigned char(double(255) + (-percent*(double(neg.b) - double(255))));
			//unsigned char a = unsigned char(double(255) + (-percent*(double(neg.a) - double(255))));
			buffer[col * m + row] = ((r << 16) | ((g << 8) | b));
		}
	}
}
void d_drawPixels(int * buffer, int m,int k, const double* vals, bool discrete){ 
	Color pos = Color(100, 167, 211, 255);
	Color neg = Color(255, 184, 113, 255);
	DrawPixelsKernel << <dimGrid(m, k), dimBlock() >> >
		(buffer, m, vals, discrete, neg, pos);
}


__global__ void SigmoidKernal(double *dst, int N) {
	int tid = blockIdx.x;
	if(tid < N)
		dst[tid] = 1/(1+exp(-dst[tid]));
}

__global__ void TanhKernal(double *dst, int N) {
		int tid = blockIdx.x;
		if(tid < N)
			dst[tid] = tanh(dst[tid]);
}

__global__ void ReLUKernal(double *dst, int N) {
	int tid = blockIdx.x;
	if(tid < N)
		dst[tid] = max(0.f, dst[tid]);
}

__global__ void LReLUKernal(double *dst, int N) {
	int tid = blockIdx.x;
	if(tid < N)
		dst[tid] = max(dst[tid] * LRELU_LEAK, dst[tid]);
}

void d_Activate(d_Matrix *dst, Activation act) {
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

__global__ void BackSigmoidKernel(double *dst, double *d_W, double *d_dZ, const double *d_A, int m, int n, int k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(col < k && row < m) {
		double tempSum = 0.0;
		for(int ind = 0; ind < n; ++ind) {
			tempSum += d_W[row * n + ind] * d_dZ[col * n + ind];
		}
		dst[col * m + row] = tempSum * (1 - d_A[col * m + row]);
	}
} /* dst = (d_W.T * d_dZ) (*) d_A^2 */
void d_BackSigmoid(d_Matrix *dst, d_Matrix *d_W, d_Matrix *d_dZ, d_Matrix *d_A) {
	int m = d_W->cols(); //reverse for transpose
	int n = d_W->rows(); //reverse for transpose
	int k = d_dZ->cols();
	BackSigmoidKernel << <dimGrid(m, k), dimBlock() >> > (dst->d_data(), d_W->d_data(), d_dZ->d_data(), d_A->d_data(), m, n, k);
}

__global__ void BackTanhKernel(double *dst, double *d_W, double *d_dZ, const double *d_A, int m, int n, int k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(col < k && row < m) {
		double tempSum = 0.0;
		for(int ind = 0; ind < n; ++ind) {
			tempSum += d_W[row * n + ind] * d_dZ[col * n + ind];
		}
		dst[col * m + row] = tempSum * (1 - d_A[col * m + row] * d_A[col * m + row]);
	}
} /* dst = (d_W.T * d_dZ) (*) d_A^2 */
void d_BackTanh(d_Matrix *dst, d_Matrix *d_W, d_Matrix *d_dZ, d_Matrix *d_A) {
	int m = d_W->cols(); //reverse for transpose
	int n = d_W->rows(); //reverse for transpose
	int k = d_dZ->cols();
	BackTanhKernel << <dimGrid(m, k), dimBlock() >> > (dst->d_data(), d_W->d_data(), d_dZ->d_data(), d_A->d_data(), m, n, k);
}

__global__ void BackReLUKernel(double *dst, double *d_W, double *d_dZ, const double *d_A, int m, int n, int k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(col < k && row < m) {
		double tempSum = 0.0;
		for(int ind = 0; ind < n; ++ind) {
			tempSum += d_W[row * n + ind] * d_dZ[col * n + ind];
		}
		dst[col * m + row] = tempSum *(d_A[col * m + row] > 0.f ? 1.f : 0.f);
	}
} /* dst = (d_W.T * d_dZ) (*) (d_A > 0 ? 1 : 0) */
void d_BackReLU(d_Matrix *dst, d_Matrix *d_W, d_Matrix *d_dZ, d_Matrix *d_A) {
	int m = d_W->cols(); //reverse for transpose
	int n = d_W->rows(); //reverse for transpose
	int k = d_dZ->cols();
	BackReLUKernel << <dimGrid(m, k), dimBlock() >> > 
		(dst->d_data(), d_W->d_data(), d_dZ->d_data(), d_A->d_data(), m, n, k);
}

__global__ void BackLReLUKernel(double *dst, double *d_W, double *d_dZ, const double *d_A, int m, int n, int k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(col < k && row < m) {
		double tempSum = 0.0;
		for(int ind = 0; ind < n; ++ind) {
			tempSum += d_W[row * n + ind] * d_dZ[col * n + ind];
		}
		dst[col * m + row] = tempSum *(d_A[col * m + row] > 0.f ? 1.f : LRELU_LEAK);
	}
} /* dst = (d_W.T * d_dZ) (*) (d_A > 0 ? 1 : 0) */
void d_BackLReLU(d_Matrix *dst, d_Matrix *d_W, d_Matrix *d_dZ, d_Matrix *d_A) {
	int m = d_W->cols(); //reverse for transpose
	int n = d_W->rows(); //reverse for transpose
	int k = d_dZ->cols();
	BackLReLUKernel << <dimGrid(m, k), dimBlock() >> >
		(dst->d_data(), d_W->d_data(), d_dZ->d_data(), d_A->d_data(), m, n, k);
}

__global__ void Set_dW_Kernel(double *dst, const double *d_dZ, const double *d_A, double coefficient, int m, int n, int k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(col < k && row < m) {
		double tempSum = 0.0;
		for(int ind = 0; ind < n; ++ind) {
			tempSum += d_dZ[row + m * ind] * d_A[col + k * ind];
		}
		dst[col * m + row] = tempSum * coefficient;
	}
} /* dst = coeff * (d_dZ * d_A.T) */
void d_Set_dW(d_Matrix* dst, d_Matrix* d_dZ, d_Matrix* d_A, double coefficient) {
	int m = d_dZ->rows();
	int n = d_dZ->cols();
	int k = d_A->rows();
	Set_dW_Kernel << <dimGrid(m, k), dimBlock() >> >
		(dst->d_data(), d_dZ->d_data(), d_A->d_data(), coefficient, m, n, k);
}

//coeff * MatrixXd((dZ * lowerA.transpose()).array() + (0.5f * (trainParams.regTerm*trainParams.learningMod) * network->GetParams().W[l]).array())
__global__ void Set_dW_Kernel(double *dst, const double *d_dZ, const double *d_A, const double *d_W, double coefficient, double regTerm, int m, int n, int k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(col < k && row < m) {
		double tempSum = 0.0;
		for(int ind = 0; ind < n; ++ind) {
			tempSum += d_dZ[row + m * ind] * d_A[col + k * ind];
		}
		//dst[col * m + row] = (coefficient * tempSum) + (coefficient * regTerm * d_W[col * m + row]); //TODO: why is this better??
		dst[col * m + row] = coefficient * (tempSum + (0.5 * regTerm * d_W[col * m + row])); 
	}
} /* dst = coeff * (d_dZ * d_A.T) (+) (0.5 * learn * d_W) */
void d_Set_dW(d_Matrix* dst, d_Matrix* d_dZ, d_Matrix* d_A, d_Matrix *d_W, double coefficient, double regTerm) {
	int m = d_dZ->rows();
	int n = d_dZ->cols();
	int k = d_A->rows();
	Set_dW_Kernel << <dimGrid(m, k), dimBlock() >> >
		(dst->d_data(), d_dZ->d_data(), d_A->d_data(), d_W->d_data(), coefficient, regTerm, m, n, k);
}

__global__ void Set_db_Kernel(double *dst, const double *d_dZ, double coefficient, int r, int c) {
	int tid = blockIdx.x;
	if(tid < r*c){ 
		double tempSum = 0.0;
		for(int ind = 0; ind < c; ++ind) {
			tempSum += d_dZ[r * ind];
		}
	dst[tid] = tempSum * coefficient;
	}
} /* dst = coeff * (srcA.SumOfRows) */
void d_Set_db(d_Matrix* dst, d_Matrix* d_dZ, double coefficient) {
	Set_db_Kernel << <d_dZ->rows(), 1 >> >
		(dst->d_data(), d_dZ->d_data(), coefficient, d_dZ->rows(), d_dZ->cols());
}