#pragma once

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class d_MatrixXf {
public:
	d_MatrixXf();
	d_MatrixXf(float * host_data, int rows, int cols_);
	~d_MatrixXf();
	float* d_data() {
		return device_data;
	}
	int rows() {
		return rowCount;
	}
	int cols() {
		return colCount;
	}
	int size() {
		return rowCount * colCount;
	}
	size_t memSize() {
		return size() * sizeof(float);
	}
	void free() {
		cudaFree(device_data);
	}

	protected:
	int rowCount;
	int colCount;
	float* device_data;
};
