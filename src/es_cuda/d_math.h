#pragma once
#include "d_cudahelpers.h"
#include "types.h"
#include "color.h"
#include "d_Matrix.h"
#include <vector>
using namespace std;
#define BLOCK_SIZE 16
#define LRELU_LEAK 0.01f
inline dim3 dimGrid(int m, int k) {
	return dim3((k + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);
}
inline dim3 dimBlock() {
	return dim3(BLOCK_SIZE, BLOCK_SIZE);
}

/* dst = srcA (+) srcB */
void d_add_elem(d_Matrix *dst, d_Matrix *srcA, d_Matrix *srcB);
/* dst = srcA (-) srcB */
void d_subtract_elem(d_Matrix *dst, d_Matrix *srcA, d_Matrix *srcB);
/* dst = srcA * b */
void d_mult_scalar(d_Matrix *dst, float b);
/* dst = src.T */
void d_transpose(d_Matrix *dst, d_Matrix *src);
/* dst = srcA * srcB */
void d_mult(d_Matrix * dst, d_Matrix * srcA, d_Matrix * srcB);
/* dst = srcA.T * srcB */
void d_mult_lhsT(d_Matrix * dst, d_Matrix * srcA, d_Matrix * srcB);
/* dst = srcA * srcB.T */
void d_mult_rhsT(d_Matrix * dst, d_Matrix * srcA, d_Matrix * srcB);
/* dst = src.sum() */
void d_sum(float *dst, d_Matrix* src);
void d_sumMatrix(float* dst, d_Matrix* src);
void d_sumMatrix(float* dst, float* src, int m, int k);
void d_square(float *dst, d_Matrix* src);
void d_square(d_Matrix* dst, d_Matrix* src);
void d_forwardLayer(d_Matrix *dst, d_Matrix *d_W, d_Matrix *d_last, d_Matrix *d_bias);
void d_activate(d_Matrix *dst, Activation act);
void d_backSigmoid(d_Matrix *dst, d_Matrix *d_W, d_Matrix *d_dZ, d_Matrix *d_A);
void d_backTanh(d_Matrix *dst, d_Matrix *d_W, d_Matrix *d_dZ, d_Matrix *d_A);
void d_backReLU(d_Matrix *dst, d_Matrix *d_W, d_Matrix *d_dZ, d_Matrix *d_A);
void d_backLReLU(d_Matrix *dst, d_Matrix *d_W, d_Matrix *d_dZ, d_Matrix *d_A);
void d_backSine(d_Matrix *dst, d_Matrix *d_W, d_Matrix *d_dZ, d_Matrix *d_A);
void d_set_dW(d_Matrix *dst, d_Matrix *d_dZ, d_Matrix *d_A, float coefficient);
void d_set_dW_Reg(d_Matrix *dst, d_Matrix *d_dZ, d_Matrix *d_A, d_Matrix *d_W, float coefficient, float regTerm);
void d_set_db(d_Matrix *dst, d_Matrix *d_dZ, float coefficient);
void d_updateParameterADAM(d_Matrix * dst, d_Matrix * d_derivative, d_Matrix * d_momentum, d_Matrix * d_momentumSqr, float learnRate);
void d_updateParameter(d_Matrix * dst, d_Matrix * d_derivative, float learnRate);
void d_calcCost(float *dst, d_Matrix* d_modelErr, vector<d_Matrix>* d_modelWeights, float regMult, float coeff, float trainLabelCount);
void d_drawPixels(int * buffer, int m, int k, const float * vals, bool discrete);