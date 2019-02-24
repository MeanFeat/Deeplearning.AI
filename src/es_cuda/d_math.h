#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "types.h"
#include "color.h"
#include "d_Matrix.h"

#define BLOCK_SIZE 32
#define LRELU_LEAK 0.01

void d_add(d_Matrix *dst, d_Matrix *srcA, d_Matrix *srcB);
void d_subtract(d_Matrix *dst, d_Matrix *srcA, d_Matrix *srcB);
void d_mult(d_Matrix * dst, d_Matrix * srcA, d_Matrix * srcB);
void d_mult_lhsT(d_Matrix * dst, d_Matrix * srcA, d_Matrix * srcB);
void d_mult_rhsT(d_Matrix * dst, d_Matrix * srcA, d_Matrix * srcB); 
void d_sum(double *dst, d_Matrix* src);
void d_square(double *dst, d_Matrix* src);
void d_squareSum(double *dst, d_Matrix* src);
void d_forwardLayer(d_Matrix *dst, d_Matrix *d_W, d_Matrix *d_last, d_Matrix *d_bias);
void d_Activate(d_Matrix *dst, Activation act);
void d_backSigmoid(d_Matrix *dst, d_Matrix *d_W, d_Matrix *d_dZ, d_Matrix *d_A);
void d_backTanh(d_Matrix *dst, d_Matrix *d_W, d_Matrix *d_dZ, d_Matrix *d_A);
void d_backReLU(d_Matrix *dst, d_Matrix *d_W, d_Matrix *d_dZ, d_Matrix *d_A);
void d_backLReLU(d_Matrix *dst, d_Matrix *d_W, d_Matrix *d_dZ, d_Matrix *d_A);
void d_set_dW(d_Matrix *dst, d_Matrix *d_dZ, d_Matrix *d_A, double coefficient);
void d_set_dW(d_Matrix *dst, d_Matrix *d_dZ, d_Matrix *d_A, d_Matrix *d_W, double coefficient, double learn);
void d_set_db(d_Matrix *dst, d_Matrix *d_dZ, double coefficient);
void d_updateParameterADAM(d_Matrix * dst, d_Matrix * d_derivative, d_Matrix * d_momentum, d_Matrix * d_momentumSqr, double learnRate);
void d_updateParameter(d_Matrix * dst, d_Matrix * d_derivative, double learnRate);

void d_calcCost(double *dst, d_Matrix* d_modelErr, double coeff);
void d_drawPixels(int * buffer, int m, int k, const double * vals, bool discrete);