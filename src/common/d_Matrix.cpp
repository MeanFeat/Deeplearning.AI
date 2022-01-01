#include "d_Matrix.h"
#include "..\es_cuda\d_math.h"
d_Matrix::d_Matrix() {}
d_Matrix::d_Matrix(int rows, int cols) {
	this->rowCount = rows;
	this->colCount = cols;
	d_check(cudaMalloc((void**)&device_data, memSize()));
}
d_Matrix::d_Matrix(float *host_data, int rows, int cols) {
	this->rowCount = rows;
	this->colCount = cols;
	d_check(cudaMalloc((void**)&device_data, memSize()));
	d_check(cudaMemcpy(device_data, host_data, memSize(), cudaMemcpyHostToDevice));
}
d_Matrix::~d_Matrix() {
	/*cudaPointerAttributes attr = {};
	cudaPointerGetAttributes(&attr, device_data);
	if (attr.devicePointer != NULL) {
		d_check(cudaFree(device_data));
	}*/
}

d_Matrix d_Matrix::serialize() {
	d_Matrix result = getClone();
	result.serializeInPlace();
	return result;
}

d_Matrix d_Matrix::serialize() const {
	d_Matrix result = getClone();
	result.serializeInPlace();
	return result;
}

void d_Matrix::serializeInPlace() {
	colCount = size();
	rowCount = 1;
}

void d_Matrix::free() {
	d_check(cudaFree(device_data));
}

d_Matrix d_Matrix::getClone() {
	d_Matrix result = d_Matrix(rowCount, colCount);
	cudaMemcpy((void**)result.d_data(), device_data, memSize(), cudaMemcpyDeviceToDevice);
	return result;
}
d_Matrix d_Matrix::getClone() const {
	d_Matrix const& result = d_Matrix(rowCount, colCount);
	cudaMemcpy((void**)result.d_data(), device_data, memSize(), cudaMemcpyDeviceToDevice);
	return result;
}