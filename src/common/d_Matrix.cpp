#include "d_Matrix.h"

d_MatrixXf::d_MatrixXf() {
}

d_MatrixXf::d_MatrixXf(float * host_data, int rows, int cols) {
	this->rowCount = rows;
	this->colCount = cols;
	cudaMalloc((void **)&device_data, memSize());
	cudaMemcpy(device_data, host_data, memSize(), cudaMemcpyHostToDevice);
}

d_MatrixXf::~d_MatrixXf() {
	//cudaFree(device_data);
}
