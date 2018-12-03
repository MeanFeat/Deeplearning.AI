#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


__global__ void addKernel(int *c, const int *a, const int *b);
void d_forwardLayer(float* d_Z, float* d_last, float* d_W, float* d_b, int midDim, int lfsRows, int rhsCols);
void d_Activate(float* dst, int size, int act);