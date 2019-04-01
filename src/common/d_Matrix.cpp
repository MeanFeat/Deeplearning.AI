#include "d_Matrix.h"

d_Matrix::d_Matrix() {
}

d_Matrix::d_Matrix(double *host_data, int rows, int cols) {
	this->rowCount = rows;
	this->colCount = cols;
	d_check(cudaMalloc((void **)&device_data, memSize()));
	d_check(cudaMemcpy(device_data, host_data, memSize(), cudaMemcpyHostToDevice));
}

d_Matrix::~d_Matrix() {
	//cudaFree(device_data);
}
