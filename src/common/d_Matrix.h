#pragma once

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class d_Matrix {
public:
	d_Matrix();
	d_Matrix(double *host_data, int rows, int cols_);
	~d_Matrix();
	double* d_data() {
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
		return size() * sizeof(double);
	}
	void free() {
		cudaFree(device_data);
	}

	protected:
	int rowCount;
	int colCount;
	double* device_data;
};
