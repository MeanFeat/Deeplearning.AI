#include "d_Matrix.h"

d_MatrixXf::d_MatrixXf() {
}

d_MatrixXf::d_MatrixXf(MatrixXf m) {
	h_mat = m;
	cudaMalloc((void **)&device_data, h_mat.size() * sizeof(float));
	cudaMemcpy(device_data, h_mat.data(), h_mat.size() * sizeof(float), cudaMemcpyHostToDevice);
}

d_MatrixXf::~d_MatrixXf() {
}
