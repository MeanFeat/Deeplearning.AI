#pragma once

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "d_math.h"
#include <Eigen/dense>

#define BLOCK_SIZE 32

using namespace Eigen;

class d_MatrixXf {
public:
	d_MatrixXf();
	d_MatrixXf(MatrixXf m);
	~d_MatrixXf();
	float* d_data() {
		return device_data;
	}
	int rows() {
		return (int)h_mat.rows();
	}
	int cols() {
		return (int)h_mat.cols();
	}
	int size() {
		return (int)h_mat.size();
	}
	size_t memSize() {
		return size() * sizeof(float);
	}
	MatrixXf h_matrix() {
		return h_mat;
	}
	void UpdateHostData() {
		cudaMemcpy(h_mat.data(), device_data, memSize(), cudaMemcpyDeviceToHost);
	}

private:
	MatrixXf h_mat;
	float* device_data;
};
