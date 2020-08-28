#include "d_math.h"

using namespace std;
__device__ ptrFunc d_pfAdd = __fadd_rn;
__device__ ptrFunc d_pfSub = __fsub_rn;
__device__ ptrFunc d_pfMult = __fmul_rn;
__device__ ptrFunc d_pfSet = _set_elem;
static ptrFunc pfAdd;
static ptrFunc pfSub;
static ptrFunc pfMult;
static ptrFunc pfSet;

#define setFunctionPointer( h_ptr, d_ptr ) cudaMemcpyFromSymbol(&h_ptr, d_ptr, sizeof(ptrFunc));

void d_mathInit() {
	if (!isInitialized) {
		cublasCreate(&cublasHandle); d_catchErr();
		setFunctionPointer(pfAdd, d_pfAdd);
		setFunctionPointer(pfSub, d_pfSub);
		setFunctionPointer(pfMult, d_pfMult);
		setFunctionPointer(pfSet, d_pfSet);
	}
}
__global__
void launch2D_elem_Kernel(ptrFunc op, float *c, const float *a, const float *b, int m, int k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = col * m + row;
	if (col < k && row < m) {
		c[tid] = (*op)(a[tid], b[tid]);
	}
}
__global__
void launch_elem_broad_Kernel(ptrFunc op, float *c, const float *a, const float b, int m, int k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = col * m + row;
	if (col < k && row < m) {
		c[tid] = (*op)(a[tid], b);
	}
}
/* dst (op)= a */
void d_launch_single_thread(ptrFunc op, float *dst, const float a) {
	launch_elem_broad_Kernel << <1, 1 >> > (op, dst, dst, a, 1, 1);
}
/* dst (op)= srcA */
void d_launch_single_thread(ptrFunc op, float *dst, const float *srcA) {
	launch2D_elem_Kernel << <1, 1 >> > (op, dst, dst, srcA, 1, 1);
}
/* dst = srcA (op) b */
void d_launch_single_thread(ptrFunc op, float *dst, const float *srcA, const float b) {
	launch_elem_broad_Kernel << <1, 1 >> > (op, dst, srcA, b, 1, 1);
}
/* dst = srcA (op) srcB */
void d_launch_single_thread(ptrFunc op, float *dst, const float *srcA, const float *srcB) {
	launch2D_elem_Kernel << <1, 1 >> > (op, dst, srcA, srcB, 1, 1);
}
/* dst = srcA (op) srcB */
void d_launch2D_elem(ptrFunc func, d_Matrix *dst, const float *srcA, const float *srcB) {
	int m = dst->rows();
	int k = dst->cols();
	launch2D_elem_Kernel << <dimGrid(m, k), dimBlock() >> > (func, dst->d_data(), srcA, srcB, m, k);
	d_catchErr();
}
/* dst = srcA (op) b */
void d_launch2D_elem(ptrFunc func, d_Matrix *dst, const float * srcA, const float b) {
	int m = dst->rows();
	int k = dst->cols();
	launch_elem_broad_Kernel << <dimGrid(m, k), dimBlock() >> > (func, dst->d_data(), srcA, b, m, k);
	d_catchErr();
}
/* dst = a */
void d_set_elem(float *dst, const float a) {
	d_launch_single_thread(pfSet, dst, a);
}
/* dst = a */
void d_set_elem(d_Matrix *dst, const float a) {
	int m = dst->rows();
	int k = dst->cols();
	launch_elem_broad_Kernel << <dimGrid(m, k), dimBlock() >> > (pfSet, dst->d_data(), dst->d_data(), a, m, k);
}
/* dst = srcA (+) srcB */
void d_add_elem(d_Matrix *dst, const d_Matrix &srcA, const d_Matrix &srcB) {
	d_launch2D_elem(pfAdd, dst, srcA.d_data(), srcB.d_data());
	d_catchErr();
}
/* dst = srcA (-) srcB */
void d_subtract_elem(d_Matrix *dst, const d_Matrix &srcA, const d_Matrix &srcB) {
	d_launch2D_elem(pfSub, dst, srcA.d_data(), srcB.d_data());
	d_catchErr();
}
/* dst = srcA (*) srcB */
void d_mult_elem(d_Matrix *dst, const d_Matrix &srcA, const d_Matrix &srcB) {
	d_launch2D_elem(pfMult, dst, srcA.d_data(), srcB.d_data());
	d_catchErr();
}
__global__
void mult_scalar_Kernel(float *dst, const float b, int m, int k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = col * m + row;
	if (col < k && row < m) {
		dst[tid] = dst[tid] * b;
	}
}
void d_mult_scalar(float *dst, const float b, int m, int k) {
	mult_scalar_Kernel << <dimGrid(m, k), dimBlock() >> > (dst, b, m, k);
	d_catchErr();
}
void d_mult_scalar(d_Matrix *dst, const float b) {
	int m = dst->rows();
	int k = dst->cols();
	d_mult_scalar(dst->d_data(), b, m, k);
	d_catchErr();
}
__global__
void transpose_Kernel(float *dst, const float *src, int m, int k) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	if (row < m && col < k) {
		dst[col + row * k] = src[row + col * m];
	}
} /* dst = src.T */
void d_transpose(d_Matrix *dst, const d_Matrix *src) {
	int c = src->cols();
	int r = src->rows();
	const float alpha = 1.f;
	const float beta = 0.f;
	cublasSgeam(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_T,
		c, r, &alpha,
		src->d_data(), r,
		&beta,
		src->d_data(), r,
		dst->d_data(), c);
	d_catchErr();
} /* dst = srcA * srcB */
void d_mult(d_Matrix* dst, const d_Matrix* srcA, const d_Matrix* srcB) {
	const float alpha = 1.f;
	const float beta = 0.f;
	const int m = srcA->rows();
	const int n = srcB->cols();
	const int k = srcA->cols();
	cublasSgemm(cublasHandle,
		CUBLAS_OP_N, CUBLAS_OP_N,
		m, n, k,
		&alpha,
		srcA->d_data(), m,
		srcB->d_data(), k,
		&beta,
		dst->d_data(), m);
	d_catchErr();
} /*dst = srcA.T * srcB */
void d_mult_lhsT(d_Matrix* dst, const d_Matrix* srcA, const d_Matrix* srcB) {
	d_Matrix d_trans = d_Matrix(srcA->cols(), srcA->rows());
	d_transpose(&d_trans, srcA); d_catchErr();
	d_mult(dst, &d_trans, srcB); d_catchErr();
	d_trans.free();
} /* dst = srcA * srcB.T */
void d_mult_rhsT(d_Matrix* dst, const d_Matrix *srcA, const d_Matrix *srcB) {
	d_Matrix d_trans = d_Matrix(srcB->cols(), srcB->rows());
	d_transpose(&d_trans, srcB); d_catchErr();
	d_mult(dst, srcA, &d_trans); d_catchErr();
	d_trans.free();
}
void d_sumMatrix(float* dst, const d_Matrix *src) {
	if (src->size() < 9999999) {
		d_Matrix serial = src->serialize(); d_catchErr();
		d_Matrix ones = d_Matrix(src->size(), 1);
		d_set_elem(&ones, 1.f); d_catchErr();
		d_Matrix result = d_Matrix(1, 1);
		d_mult(&result, &serial, &ones); d_catchErr();
		cudaMemcpy((void**)dst, result.d_data(), sizeof(float), cudaMemcpyDeviceToDevice);
		serial.free();
		ones.free();
		result.free();
	}
	else {
		//recursion
		d_Matrix r = d_Matrix(src->rows(), 1);
		d_sumRows(&r, src);
		d_sumMatrix(dst, &r);
		r.free();
	}
}
__global__
void add_row_broad_Kernel(float *dst, const float *srcMat, const float *srcVec, int m, int k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (col < k && row < m) {
		dst[col * m + row] = srcMat[col * m + row] + srcVec[row];
	}
}
/* dst = d_W * d_last + d_bias */
void d_forwardLayer(d_Matrix *dst, const d_Matrix *d_W, const d_Matrix *d_last, const d_Matrix *d_bias) {
	int m = d_W->rows();
	int n = d_W->cols();
	int k = d_last->cols();
	d_mult(dst, d_W, d_last);
	add_row_broad_Kernel << <dimGrid(m, k), dimBlock() >> > (dst->d_data(), dst->d_data(), d_bias->d_data(), m, k);
	d_catchErr();
}
__global__
void drawPixels_Kernel(int *buffer, int m, const float* vals, bool discrete, const Color neg, const Color pos) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	float percent = vals[col * m + row];
	if (discrete) {
		if (percent > 0.f) {
			buffer[col * m + row] = ((pos.r << 16) | ((pos.g << 8) | pos.b));
		}
		else {
			buffer[col * m + row] = ((neg.r << 16) | ((neg.g << 8) | neg.b));
		}
	}
	else {
		if (percent > 0.) {
			unsigned char r = unsigned char(float(255) + (percent*(float(pos.r) - float(255))));
			unsigned char g = unsigned char(float(255) + (percent*(float(pos.g) - float(255))));
			unsigned char b = unsigned char(float(255) + (percent*(float(pos.b) - float(255))));
			//unsigned char a = unsigned char(float(255) + (percent*(float(pos.a) - float(255))));
			buffer[col * m + row] = ((r << 16) | ((g << 8) | b));
		}
		else {
			unsigned char r = unsigned char(float(255) + (-percent * (float(neg.r) - float(255))));
			unsigned char g = unsigned char(float(255) + (-percent * (float(neg.g) - float(255))));
			unsigned char b = unsigned char(float(255) + (-percent * (float(neg.b) - float(255))));
			//unsigned char a = unsigned char(float(255) + (-percent*(float(neg.a) - float(255))));
			buffer[col * m + row] = ((r << 16) | ((g << 8) | b));
		}
	}
}
void d_drawPixels(int * buffer, int m, int k, const float* vals, bool discrete) {
	Color pos = Color(100, 167, 211, 255);
	Color neg = Color(255, 184, 113, 255);
	drawPixels_Kernel << <dimGrid(m, k), dimBlock() >> >
		(buffer, m, vals, discrete, neg, pos);
	d_catchErr();
}
__global__
void Sigmoid_Kernal(float *dst, int m, int k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = col * m + row;
	if (col < k && row < m) {
		dst[tid] = 1.f / (1.f + exp(-dst[tid]));
	}
}
__global__
void Tanh_Kernal(float *dst, int m, int k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = col * m + row;
	if (col < k && row < m)
		dst[tid] = tanhf(dst[tid]);
}
__global__
void ReLU_Kernal(float *dst, int m, int k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = col * m + row;
	if (col < k && row < m)
		dst[tid] = fmaxf(0.f, dst[tid]);
}
__global__
void LReLU_Kernal(float *dst, int m, int k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = col * m + row;
	if (col < k && row < m)
		dst[tid] = fmaxf(dst[tid] * LRELU_LEAK, dst[tid]);
}
__global__
void Sine_Kernal(float *dst, int m, int k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = col * m + row;
	if (col < k && row < m) {
		dst[tid] = __sinf(dst[tid]);
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
__global__
void backSigmoid_Kernel(float *dst, const float *d_A, int m, int n, int k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (col < k && row < m) {
		dst[col * m + row] *= (1 - d_A[col * m + row]);
	}
} /* dst = (d_W.T * d_dZ) (*) d_A */
void d_backSigmoid(d_Matrix *dst, const d_Matrix *d_W, const d_Matrix *d_dZ, const d_Matrix *d_A) {
	d_mult_lhsT(dst, d_W, d_dZ);
	int m = d_W->cols(); //reverse for transpose
	int n = d_W->rows(); //reverse for transpose
	int k = d_dZ->cols();
	backSigmoid_Kernel << <dimGrid(m, k), dimBlock() >> >
		(dst->d_data(), d_A->d_data(), m, n, k);
	d_catchErr();
}
__global__
void backTanh_Kernel(float *dst, const float *d_A, int m, int n, int k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (col < k && row < m) {
		dst[col * m + row] *= (1 - (d_A[col * m + row] * d_A[col * m + row]));
	}
} /* dst = (d_W.T * d_dZ) (*) 1 - d_A^2 */
void d_backTanh(d_Matrix *dst, const d_Matrix *d_W, const d_Matrix *d_dZ, const d_Matrix *d_A) {
	d_mult_lhsT(dst, d_W, d_dZ);
	int m = d_W->cols(); //reverse for transpose
	int n = d_W->rows(); //reverse for transpose
	int k = d_dZ->cols();
	backTanh_Kernel << <dimGrid(m, k), dimBlock() >> >
		(dst->d_data(), d_A->d_data(), m, n, k);
	d_catchErr();
}
__global__
void backReLU_Kernel(float *dst, const float *d_A, int m, int n, int k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (col < k && row < m) {
		dst[row * k + col] *= (d_A[row * k + col] > 0.f ? 1.f : 0.f);
	}
} /* dst = (d_W.T * d_dZ) (*) (d_A > 0 ? 1 : 0) */
void d_backReLU(d_Matrix *dst, const d_Matrix *d_W, const d_Matrix *d_dZ, const d_Matrix *d_A) {
	d_mult_lhsT(dst, d_W, d_dZ);
	int m = d_W->cols(); //reverse for transpose
	int n = d_W->rows(); //reverse for transpose
	int k = d_dZ->cols();
	backReLU_Kernel << <dimGrid(m, k), dimBlock() >> >
		(dst->d_data(), d_A->d_data(), m, n, k);
	d_catchErr();
}
__global__
void backLReLU_Kernel(float *dst, const float *d_A, int m, int n, int k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (col < k && row < m) {
		dst[row * k + col] *= (d_A[row * k + col] > 0.f ? 1.f : LRELU_LEAK);
	}
} /* dst = (d_W.T * d_dZ) (*) (d_A > 0 ? 1 : 0) */
void d_backLReLU(d_Matrix *dst, const d_Matrix *d_W, const d_Matrix *d_dZ, const d_Matrix *d_A) {
	d_mult_lhsT(dst, d_W, d_dZ);
	int m = d_W->cols(); //reverse for transpose
	int n = d_W->rows(); //reverse for transpose
	int k = d_dZ->cols();
	backLReLU_Kernel << <dimGrid(m, k), dimBlock() >> >
		(dst->d_data(), d_A->d_data(), m, n, k);
	d_catchErr();
}
__global__
void backSine_Kernel(float *dst, const float *d_A, int m, int n, int k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (col < k && row < m) {
		dst[row * k + col] *= cos(d_A[row * k + col]);
	}
} /* dst = cos(d_W.T * d_dZ) */
void d_backSine(d_Matrix *dst, const d_Matrix *d_W, const d_Matrix *d_dZ, const d_Matrix *d_A) {
	d_mult_lhsT(dst, d_W, d_dZ);
	int m = d_W->cols(); //reverse for transpose
	int n = d_W->rows(); //reverse for transpose
	int k = d_dZ->cols();
	backSine_Kernel << <dimGrid(m, k), dimBlock() >> >
		(dst->d_data(), d_A->d_data(), m, n, k);
	d_catchErr();
} /* dst = coeff * (d_dZ * d_A.T) */
void d_set_dW(d_Matrix* dst, const d_Matrix* d_dZ, const d_Matrix* d_AT, float coefficient) {
	d_mult(dst, d_dZ, d_AT); d_catchErr();
	d_mult_scalar(dst, coefficient); d_catchErr();
}
__global__
void set_dW_Reg_Kernel(float *dst, const float *d_W, float coefficient, float regTerm, int m, int n, int k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (col < k && row < m) {
		dst[col * m + row] += (regTerm * d_W[col * m + row]);
	}
} /* dst = coeff * (d_dZ * d_A.T) (+) (0.5f * learn * d_W) */
void d_set_dW_Reg(d_Matrix* dst, const d_Matrix* d_dZ, const d_Matrix* d_AT, const d_Matrix *d_W, float coefficient, float regTerm) {
	int m = d_dZ->rows();
	int n = d_dZ->cols();
	int k = d_AT->cols();
	d_mult(dst, d_dZ, d_AT);
	set_dW_Reg_Kernel << <dimGrid(m, k), dimBlock() >> > (dst->d_data(), d_W->d_data(), coefficient, regTerm, m, n, k);
	d_mult_scalar(dst, coefficient);
	d_catchErr();
}
void d_sumRows(d_Matrix* dst, const d_Matrix* src) {
	int m = src->rows();
	int k = src->cols();
	d_Matrix ones = d_Matrix(k, 1);
	d_set_elem(&ones, 1.f);
	d_mult(dst, src, &ones);
	ones.free();
}
/* dst = coeff * (srcA.SumOfRows) */
void d_set_db(d_Matrix* dst, const d_Matrix* d_dZ, float coefficient) {
	int m = d_dZ->rows();
	int k = d_dZ->cols();
	d_sumRows(dst, d_dZ);
	d_mult_scalar(dst, coefficient);
	d_catchErr();
}
#define BETA1 0.9f
#define BETA2 (1.f - FLT_EPSILON)
__global__
void updateParameterADAM_Kernel(float *dst, int N, const float *d_derivative, float *d_momentum, float *d_momentumSqr, float learn, int k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = row * k + col;
	if (tid < N) {
		d_momentum[tid] = BETA1 * (d_momentum[tid]) + (1.f - BETA1) * d_derivative[tid];
		d_momentumSqr[tid] = (BETA2 * d_momentumSqr[tid]) + ((1.f - BETA2) * (d_derivative[tid] * d_derivative[tid]));
		dst[tid] -= learn * (d_momentum[tid] / (1.f - (BETA1 * BETA1)) / (sqrtf(d_momentumSqr[tid] / (1.f - (BETA2 * BETA2))) + FLT_EPSILON));
	}
}
void d_updateParameterADAM(d_Matrix* dst, const d_Matrix* d_derivative, const d_Matrix* d_momentum, const d_Matrix* d_momentumSqr, float learnRate) {
	int m = dst->rows();
	int k = dst->cols();
	updateParameterADAM_Kernel << <dimGrid(m, k), dimBlock() >> >
		(dst->d_data(), dst->size(), d_derivative->d_data(), d_momentum->d_data(), d_momentumSqr->d_data(), learnRate, k);
	d_catchErr();
}
__global__
void updateParameter_Kernel(float *dst, int N, const float *d_derivative, float learn) {
	int tid = blockIdx.x;
	if (tid < N) {
		dst[tid] -= learn * d_derivative[tid];
	}
}
void d_updateParameter(d_Matrix* dst, const d_Matrix* d_derivative, float learnRate) {
	updateParameter_Kernel << <dst->size(), 1 >> >
		(dst->d_data(), dst->size(), d_derivative->d_data(), learnRate);
	d_catchErr();
}
void d_square(d_Matrix* dst, const d_Matrix* src) {
	d_launch2D_elem(pfMult, dst, src->d_data(), src->d_data());
}
__global__
void finalCost_Kernel(float *dst, const float *sumTotal, float regMult, float trainCount) {
	dst[0] = dst[0] + (0.5f * (regMult * (sumTotal[0] / (trainCount*2.0f))));
}

void d_calcCost(float *dst, const d_Matrix* d_err, const vector<d_Matrix>* d_modelWeights, float regMult, float coeff, float trainLableCount) {
	d_Matrix *d_diff = new d_Matrix(d_err->rows(), d_err->cols());
	int m = d_err->rows();
	int k = d_err->cols();
	d_square(d_diff, d_err);
	d_sumMatrix(dst, d_diff);
	d_mult_scalar(dst, coeff, 1, 1);
	// Add Regularization
	d_Matrix d_sqrSumTotal = d_Matrix(1, 1);
	d_set_elem(d_sqrSumTotal.d_data(), 0.f);
	d_Matrix d_sqrSum = d_Matrix(1, 1);
	for (int i = 0; i < (int)d_modelWeights->size() - 1; ++i) {
		const d_Matrix *layerWeights = &d_modelWeights->at(i);
		d_Matrix d_squared(layerWeights->rows(), layerWeights->cols());
		d_square(&d_squared, layerWeights); d_catchErr();
		d_sumMatrix(d_sqrSum.d_data(), &d_squared); d_catchErr();
		d_launch_single_thread(pfAdd, d_sqrSumTotal.d_data(), d_sqrSum.d_data()); d_catchErr();
	}
	finalCost_Kernel << <1, 1 >> > (dst, d_sqrSumTotal.d_data(), regMult, trainLableCount); d_catchErr();
	d_sqrSumTotal.free();
	d_sqrSum.free();
}