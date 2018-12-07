#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "types.h"
#include "color.h"
#include "d_Matrix.h"

#define BLOCK_SIZE 32
#define LRELU_LEAK 0.05f

void d_add(d_MatrixXf *dst, d_MatrixXf *srcA, d_MatrixXf *srcB);
void d_subtract(d_MatrixXf *dst, d_MatrixXf *srcA, d_MatrixXf *srcB);
void d_matrixMult(d_MatrixXf* dst, d_MatrixXf* srcA, d_MatrixXf* srcB);
void d_matrixMult_lhsT(d_MatrixXf* dst, d_MatrixXf* srcA, d_MatrixXf* srcB);
void d_matrixMult_rhsT(d_MatrixXf* dst, d_MatrixXf* srcA, d_MatrixXf* srcB);

void d_forwardLayer(d_MatrixXf *dst, d_MatrixXf *d_W, d_MatrixXf *d_last, d_MatrixXf *d_bias);
void d_Activate(d_MatrixXf *dst, Activation act);
void d_BackSigmoid(d_MatrixXf *dst, d_MatrixXf *d_W, d_MatrixXf *d_dZ, d_MatrixXf *d_A);
void d_BackTanh(d_MatrixXf *dst, d_MatrixXf *d_W, d_MatrixXf *d_dZ, d_MatrixXf *d_A);
void d_BackReLU(d_MatrixXf *dst, d_MatrixXf *d_W, d_MatrixXf *d_dZ, d_MatrixXf *d_A);
void d_BackLReLU(d_MatrixXf *dst, d_MatrixXf *d_W, d_MatrixXf *d_dZ, d_MatrixXf *d_A);
void d_Set_dW(d_MatrixXf *dst, d_MatrixXf *d_dZ, d_MatrixXf *d_A, float coefficient);
void d_Set_dW(d_MatrixXf *dst, d_MatrixXf *d_dZ, d_MatrixXf *d_A, d_MatrixXf *d_W, float coefficient, float learn);
void d_Set_db(d_MatrixXf *dst, d_MatrixXf *d_dZ, float coefficient);

void d_drawPixels(int * buffer, int m, int k, const float * vals, bool discrete);