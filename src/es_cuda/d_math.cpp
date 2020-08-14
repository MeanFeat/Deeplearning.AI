#include "d_math.h"

__device__ ptrFunc d_pfAdd = __fadd_rn;
__device__ ptrFunc d_pfSub = __fsub_rn;
__device__ ptrFunc d_pfMult = __fmul_rn;
static ptrFunc pfAdd;
static ptrFunc pfSub;
static ptrFunc pfMult;

#define setFunctionPointer( h_ptr, d_ptr ) cudaMemcpyFromSymbol(&h_ptr, d_ptr, sizeof(ptrFunc));

void d_mathInit(){
	if (!isInitialized) {
		setFunctionPointer(pfAdd, d_pfAdd);
		setFunctionPointer(pfSub, d_pfSub);
		setFunctionPointer(pfMult, d_pfMult);
	}
}
__global__ void launch2D_elem_Kernel(ptrFunc op, float *c, const float *a, const float *b, int m, int k){
	int row = blockIdx.y * blockDim.y + threadIdx.y; 
	int col = blockIdx.x * blockDim.x + threadIdx.x; 
	int tid = row * k + col; 
	if (col < k && row < m) {
		c[tid] = (*op)(a[tid], b[tid]);
	}
}
void launch_single_thread(ptrFunc op, float *c, const float *a, const float *b) {
	launch2D_elem_Kernel<<<1,1>>>(op, c, a, b, 1, 1);
}
template<typename Ta, typename Tb>
void launch2D_elem(ptrFunc func, d_Matrix *dst, const Ta srcA, const Tb srcB) {
	int m = dst->rows();
	int k = dst->cols();
	launch2D_elem_Kernel << <dimGrid(m, k), dimBlock() >> > (func, dst->d_data(), srcA, srcB, m, k);
	d_catchErr();
} 
/* dst = srcA (+) srcB */
void d_add_elem(d_Matrix *dst, const d_Matrix &srcA, const d_Matrix &srcB) {
	launch2D_elem(pfAdd, dst, srcA.d_data(), srcB.d_data());
	d_catchErr();
}
/* dst = srcA (-) srcB */
void d_subtract_elem(d_Matrix *dst, const d_Matrix &srcA, const d_Matrix &srcB) {
	launch2D_elem(pfSub, dst, srcA.d_data(), srcB.d_data());
	d_catchErr();
}
/* dst = srcA (*) srcB */
void d_mult_elem(d_Matrix *dst, const d_Matrix &srcA, const d_Matrix &srcB) {
	launch2D_elem(pfMult, dst, srcA.d_data(), srcB.d_data());
	d_catchErr();
}
__global__ void mult_scalar_Kernel(float *dst, const float b, int m, int k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = row * k + col;;
	if (col < k && row < m) {
		dst[tid] = dst[tid] * b;
	}
}
void d_mult_scalar(d_Matrix *dst, const float b) {
	int m = dst->rows();
	int k = dst->cols();
	mult_scalar_Kernel << <dimGrid(m, k), dimBlock() >> > (dst->d_data(), b, m, k);
	d_catchErr();
}
__global__ void transpose_Naive_Kernel(float *dst, const float *src, int m, int k) {
	int tid = threadIdx.x;
	if (tid == 0) {
		for (int i = 0; i < k; i++) {
			for (int j = 0; j < m; ++j) {
				dst[j * k+ i] = src[j * k + i];
			}
		}
		__syncthreads();
	}
} /* dst = src.T */
void d_transpose(d_Matrix *dst, d_Matrix *src) {
	int m = src->rows();
	int k = src->cols();
	transpose_Naive_Kernel << <1,1 >> > (dst->d_data(), src->d_data(), m, k);
	d_catchErr();
}
__global__ void mult_Kernel(float *dst, const float *srcA, const float *srcB, const int m, const int n, const int k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	float sum = 0.0f;
	if (col < k && row < m) {
		for (int i = 0; i < n; i++) {
			sum += srcA[row * n + i] * srcB[i * k + col];
		}
		dst[row * k + col] = sum;
	}
} /* dst = srcA * srcB */
void d_mult(d_Matrix* dst, d_Matrix* srcA, d_Matrix* srcB) {
	int m = srcA->rows();
	int n = srcA->cols();
	int k = srcB->cols();
	mult_Kernel << <dimGrid(m, k), dimBlock() >> >
		(dst->d_data(), srcA->d_data(), srcB->d_data(), m, n, k);
	d_catchErr();
}
__global__ void mult_lhsT_Kernel(float *dst, float *srcA, float *srcB, int m, int n, int k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (col < k && row < m) {
		float sum = 0.f;
		for (int i = 0; i < n; ++i) {
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
__global__ void mult_rhsT_Kernel(float *dst, const float *srcA, const float *srcB, int m, int n, int k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	float sum = 0.0f;
	if (col < k && row < m) {
		for (int i = 0; i < n; i++) {
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
__global__ void sum_Naive_Kernel(float *dst, const float *src, int len) {
	int tid = threadIdx.x;
	if (tid == 0) {
		float sum = 0.f;
		for (int i = 0; i < len; i++) {
			sum += src[i];
		}
		__syncthreads();
		dst[0] = sum;
	}
}
__global__ void sum_Kernel(float *dst, float *src, int len) {
	int tid = threadIdx.x;
	int step = 2;
	int get = 1;
	while (get < len) {
		if (tid * step + get < len) {
			src[tid*step] += src[tid*step + get];
			__syncthreads();
		}
		step *= 2;
		get *= 2;
	}
	if (tid == 0) {
		if (len % 2 > 0) {
			src[0] += src[len];
		}
		dst[0] = src[0];
	}
}
void d_sum(float *dst, d_Matrix* src) {
	int m = src->size();
	sum_Kernel << < 1, m / 2 >> > (dst, src->d_data(), m);
	d_catchErr();
}
__global__ void sumMatrix_Kernel(float *dst, float *src, int m, int k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int len = m * k;
	unsigned int step = 2;
	unsigned int get = 1;
	unsigned int tid = col + row * k;
	if (col < k && row < m) {
		while (get < len) {
			if (tid * step + get < len) {
				src[tid*step] += src[tid * step + get];
				src[tid * step + get] = 0.f;
				__syncthreads();
			}
			step *= 2;
			get *= 2;
			__syncthreads();
		}
	}
	if (tid == 0) {
		if (len % 2 > 0) {
			src[0] += src[len];
		}
		dst[0] = src[0];
	}
}
void d_sumMatrix(float* dst, d_Matrix *src){
	d_sumMatrix(dst, src->d_data(), src->rows(), src->cols());
}
void d_sumMatrix(float* dst, float* src, int m, int k) {
	int len = m * k;
	//sumMatrix_Kernel << <dimGrid(m, k), dimBlock() >> > (dst, src->d_data(), m, k);
	sum_Naive_Kernel << <1, 1 >> > (dst, src, len);
	d_catchErr();
}
__global__ void forwardLayer_Kernel(float *dst, const float *d_W, const float *d_last, const float * d_bias, int m, int n, int k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	float sum = 0.f;
	if (col < k && row < m) {
		for (int i = 0; i < n; i++) {
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
__global__ void drawPixels_Kernel(int *buffer, int k, const float* vals, bool discrete, Color neg, Color pos) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	float percent = vals[row * k + col];
	if (discrete) {
		if (percent > 0.f) {
			buffer[row * k + col] = ((pos.r << 16) | ((pos.g << 8) | pos.b));
		}
		else {
			buffer[row * k + col] = ((neg.r << 16) | ((neg.g << 8) | neg.b));
		}
	}
	else {
		if (percent > 0.) {
			unsigned char r = unsigned char(float(255) + (percent*(float(pos.r) - float(255))));
			unsigned char g = unsigned char(float(255) + (percent*(float(pos.g) - float(255))));
			unsigned char b = unsigned char(float(255) + (percent*(float(pos.b) - float(255))));
			//unsigned char a = unsigned char(float(255) + (percent*(float(pos.a) - float(255))));
			buffer[row * k + col] = ((r << 16) | ((g << 8) | b));
		}
		else {
			unsigned char r = unsigned char(float(255) + (-percent * (float(neg.r) - float(255))));
			unsigned char g = unsigned char(float(255) + (-percent * (float(neg.g) - float(255))));
			unsigned char b = unsigned char(float(255) + (-percent * (float(neg.b) - float(255))));
			//unsigned char a = unsigned char(float(255) + (-percent*(float(neg.a) - float(255))));
			buffer[row * k + col] = ((r << 16) | ((g << 8) | b));
		}
	}
}
void d_drawPixels(int * buffer, int m, int k, const float* vals, bool discrete) {
	Color pos = Color(100, 167, 211, 255);
	Color neg = Color(255, 184, 113, 255);
	drawPixels_Kernel << <dimGrid(m, k), dimBlock() >> >
		(buffer, k, vals, discrete, neg, pos);
	d_catchErr();
}
__global__ void Sigmoid_Kernal(float *dst, int m, int k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = row * k + col;
	if (col < k && row < m) {
		dst[tid] = 1.f / (1.f + exp(-dst[tid]));
	}
}
__global__ void Tanh_Kernal(float *dst, int m, int k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = row * k + col;
	if (col < k && row < m)
		dst[tid] = tanhf(dst[tid]);
}
__global__ void ReLU_Kernal(float *dst, int m, int k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = row * k + col;
	if (col < k && row < m)
		dst[tid] = fmaxf(0.f, dst[tid]);
}
__global__ void LReLU_Kernal(float *dst, int m, int k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = row * k + col;
	if (col < k && row < m)
		dst[tid] = fmaxf(dst[tid] * LRELU_LEAK, dst[tid]);
}
__global__ void Sine_Kernal(float *dst, int m, int k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = row * k + col;
	if (col < k && row < m) {
		dst[tid] = sin(dst[tid]);
	}
}
void d_activate(d_Matrix *dst, Activation act) {
	int m = dst->rows();
	int k = dst->cols();
	switch (act) {
	case Sigmoid:
		Sigmoid_Kernal << < dimGrid(m, k), dimBlock() >> > (dst->d_data(), m, k);
		break;
	case Tanh:
		Tanh_Kernal << < dimGrid(m, k), dimBlock() >> > (dst->d_data(), m, k);
		break;
	case ReLU:
		ReLU_Kernal << < dimGrid(m, k), dimBlock() >> > (dst->d_data(), m, k);
		break;
	case LReLU:
		LReLU_Kernal << < dimGrid(m, k), dimBlock() >> > (dst->d_data(), m, k);
		break;
	case Sine:
		Sine_Kernal << < dimGrid(m, k), dimBlock() >> > (dst->d_data(), m, k);
		break;
	case Linear: //fall through
	default:
		break;
	}
}
__global__ void backSigmoid_Kernel(float *dst, float *d_W, float *d_dZ, const float *d_A, int m, int n, int k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (col < k && row < m) {
		float sum = 0.f;
		for (int i = 0; i < n; ++i) {
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
__global__ void backTanh_Kernel(float *dst, float *d_W, float *d_dZ, const float *d_A, int m, int n, int k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (col < k && row < m) {
		float sum = 0.f;
		for (int i = 0; i < n; ++i) {
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
__global__ void backReLU_Kernel(float *dst, float *d_W, float *d_dZ, const float *d_A, int m, int n, int k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (col < k && row < m) {
		float sum = 0.f;
		for (int i = 0; i < n; ++i) {
			sum += d_W[row + m * i] * d_dZ[i * k + col];
		}
		dst[row * k + col] = sum * (d_A[row * k + col] > 0.f ? 1.f : 0.f);
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
__global__ void backLReLU_Kernel(float *dst, float *d_W, float *d_dZ, const float *d_A, int m, int n, int k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (col < k && row < m) {
		float sum = 0.f;
		for (int i = 0; i < n; ++i) {
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
__global__ void backSine_Kernel(float *dst, float *d_W, float *d_dZ, const float *d_A, int m, int n, int k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (col < k && row < m) {
		float sum = 0.f;
		for (int i = 0; i < n; ++i) {
			sum += d_W[row + m * i] * d_dZ[i * k + col];
		}
		dst[row * k + col] = cosf(sum);
	}
} /* dst = cos(d_W.T * d_dZ) */
void d_backSine(d_Matrix *dst, d_Matrix *d_W, d_Matrix *d_dZ, d_Matrix *d_A) {
	int m = d_W->cols(); //reverse for transpose
	int n = d_W->rows(); //reverse for transpose
	int k = d_dZ->cols();
	backSine_Kernel << <dimGrid(m, k), dimBlock() >> >
		(dst->d_data(), d_W->d_data(), d_dZ->d_data(), d_A->d_data(), m, n, k);
	d_catchErr();
}
__global__ void set_dW_Kernel(float *dst, const float *d_dZ, const float *d_A, float coefficient, int m, int n, int k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	float sum = 0.f;
	if (col < k && row < m) {
		for (int i = 0; i < n; i++) {
			sum += d_dZ[row * n + i] * d_A[col * n + i];
		}
		dst[row * k + col] = sum * coefficient;
	}
} /* dst = coeff * (d_dZ * d_A.T) */
void d_set_dW(d_Matrix* dst, d_Matrix* d_dZ, d_Matrix* d_A, float coefficient) {
	int m = d_dZ->rows();
	int n = d_dZ->cols();
	int k = d_A->rows();
	set_dW_Kernel << <dimGrid(m, k), dimBlock() >> >
		(dst->d_data(), d_dZ->d_data(), d_A->d_data(), coefficient, m, n, k);
	d_catchErr();
}
__global__ void set_dW_Reg_Kernel(float *dst, const float *d_W, float coefficient, float regTerm, int m, int n, int k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (col < k && row < m) {
		dst[row * k + col] += (regTerm * d_W[row * k + col]);
	}
} /* dst = coeff * (d_dZ * d_A.T) (+) (0.5f * learn * d_W) */
void d_set_dW_Reg(d_Matrix* dst, d_Matrix* d_dZ, d_Matrix* d_A, d_Matrix *d_W, float coefficient, float regTerm) {
	int m = d_dZ->rows();
	int n = d_dZ->cols();
	int k = d_A->rows();
#if 0
	d_Matrix d_AT = d_Matrix(d_A->cols(), d_A->rows());
	d_transpose(&d_AT, d_A);
	d_mult(dst, d_dZ, &d_AT);
#else
	d_mult_rhsT(dst, d_dZ, d_A);
#endif
	set_dW_Reg_Kernel << <dimGrid(m, k), dimBlock() >> > (dst->d_data(), d_W->d_data(), coefficient, regTerm, m, n, k);
	d_mult_scalar(dst, coefficient);
	d_catchErr();
}
__global__ void set_db_Kernel(float *dst, const float *d_dZ, int r, int c) {
	int tid = blockIdx.x;
	if (tid < r) {
		float sum = 0.f;
		for (int ind = 0; ind < c; ++ind) {
			sum += d_dZ[tid * c + ind];
		}
		dst[tid] = sum;
	}
} /* dst = coeff * (srcA.SumOfRows) */
void d_set_db(d_Matrix* dst, d_Matrix* d_dZ, float coefficient) {
	int m = d_dZ->rows();
	int k = d_dZ->cols();
	set_db_Kernel << <dst->rows(), 1 >> > (dst->d_data(), d_dZ->d_data(), m, k);
	mult_scalar_Kernel << <dimGrid(m, k), dimBlock() >> > (dst->d_data(), coefficient, m, 1);
	d_catchErr();
}
#define BETA1 0.9f
#define BETA2 (1.f - FLT_EPSILON)
__global__ void updateParameterADAM_Kernel(float *dst, int N, const float *d_derivative, float *d_momentum, float *d_momentumSqr, float learn) {
	int tid = blockIdx.x;
	if (tid < N) {
		d_momentum[tid] = BETA1 * (d_momentum[tid]) + (1.f - BETA1) * d_derivative[tid];
		d_momentumSqr[tid] = (BETA2 * d_momentumSqr[tid]) + ((1.f - BETA2) * (d_derivative[tid] * d_derivative[tid]));
		dst[tid] -= learn * (d_momentum[tid] / (1.f - (BETA1 * BETA1)) / (sqrtf(d_momentumSqr[tid] / (1.f - (BETA2 * BETA2))) + FLT_EPSILON));
	}
}
void d_updateParameterADAM(d_Matrix* dst, d_Matrix* d_derivative, d_Matrix* d_momentum, d_Matrix* d_momentumSqr, float learnRate) {
	updateParameterADAM_Kernel << <dst->size(), 1 >> >
		(dst->d_data(), dst->size(), d_derivative->d_data(), d_momentum->d_data(), d_momentumSqr->d_data(), learnRate);
	d_catchErr();
}
__global__ void updateParameter_Kernel(float *dst, int N, const float *d_derivative, float learn) {
	int tid = blockIdx.x;
	if (tid < N) {
		dst[tid] -= learn * d_derivative[tid];
	}
}
void d_updateParameter(d_Matrix* dst, d_Matrix* d_derivative, float learnRate) {
	updateParameter_Kernel << <dst->size(), 1 >> >
		(dst->d_data(), dst->size(), d_derivative->d_data(), learnRate);
	d_catchErr();
}

__global__ void square_Kernel(float *dst, float *src, int m, int k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (col < k && row < m) {
		dst[row * k + col] = src[row * k + col] * src[row * k + col];
	}
}
__global__ void square_Kernel(float *dst, float *src) {
	int tid = threadIdx.x;
	dst[tid] = src[tid] * src[tid];
}
void d_square(float *dst, d_Matrix* src) {
	square_Kernel << <1, src->size() >> > (dst, src->d_data());
}
void d_square(d_Matrix* dst, d_Matrix* src) {
	launch2D_elem(pfMult, dst, src->d_data(), src->d_data());
}
__global__ void finalCost_Kernel(float *dst, float *sumTotal, float regMult, float trainCount) {
	dst[0] = dst[0] + (0.5f * (regMult * (sumTotal[0] / (trainCount*2.0f))));
}
__global__ void setZero_Kernel(float *dst) {
	dst[0] = 0.0f;
}
void d_calcCost(float *dst, d_Matrix* d_modelErr, vector<d_Matrix>* d_modelWeights, float regMult, float coeff, float trainLableCount) {
	float* d_diff;
	int m = d_modelErr->rows();
	int k = d_modelErr->cols();
	int s = d_modelErr->size();
	cudaMalloc((void**)&d_diff, d_modelErr->memSize());
	square_Kernel << <dimGrid(m, k), dimBlock() >> > (d_diff, d_modelErr->d_data(), m, k); d_catchErr();
	d_sumMatrix(dst, d_diff, m,k); d_catchErr();
	mult_scalar_Kernel << < 1, 1 >> > (dst, coeff, m, k); d_catchErr();
	//return;
	// Add Regularization
	float* d_sqrSumTotal;
	cudaMalloc((void**)&d_sqrSumTotal, sizeof(float));
	setZero_Kernel << <1, 1 >> > (d_sqrSumTotal); d_catchErr();
	for (int i = 0; i < (int)d_modelWeights->size() - 1; ++i) {
		int m = d_modelWeights->at(i).rows();
		int k = d_modelWeights->at(i).cols();
		float* d_squared;
		float* d_sqrSum;
		d_check(cudaMalloc((void**)&d_sqrSum, sizeof(float)));
		d_check(cudaMalloc((void**)&d_squared, d_modelWeights->at(i).memSize()));
		square_Kernel << < dimGrid(m, k), dimBlock() >> > (d_squared, d_modelWeights->at(i).d_data(), m, k); d_catchErr();
		d_sumMatrix(d_sqrSum, d_squared, m, k);
		launch_single_thread(pfAdd, d_sqrSumTotal, d_sqrSumTotal, d_sqrSum);
		d_check(cudaFree(d_sqrSum));
		d_check(cudaFree(d_squared));
	}
	finalCost_Kernel << <1, 1 >> > (dst, d_sqrSumTotal, regMult, trainLableCount); d_catchErr();
	cudaFree(d_sqrSumTotal);
	cudaFree(d_diff);
	d_catchErr();
}