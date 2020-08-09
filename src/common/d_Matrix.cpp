#include "d_Matrix.h"
d_Matrix::d_Matrix(){}
d_Matrix::d_Matrix(int rows, int cols) {
	this->rowCount = rows;
	this->colCount = cols;
}
d_Matrix::d_Matrix(float *host_data, int rows, int cols){
	this->rowCount = rows;
	this->colCount = cols;
	d_check(cudaMalloc((void**)&device_data, memSize()));
	d_check(cudaMemcpy(device_data, host_data, memSize(), cudaMemcpyHostToDevice));
}
d_Matrix::~d_Matrix(){
	//cudaFree(device_data);
}

using namespace Eigen;
d_Matrix to_device(MatrixXf matrix) {
	//transpose data only to Column Major
	MatrixXf temp = matrix.transpose();
	return d_Matrix(temp.data(), (int)matrix.rows(), (int)matrix.cols());
}
MatrixXf to_host(d_Matrix d_matrix) {
	// return to Row Major order
	MatrixXf out = MatrixXf(d_matrix.cols(), d_matrix.rows());
	d_check(cudaMemcpy(out.data(), d_matrix.d_data(), d_matrix.memSize(), cudaMemcpyDeviceToHost));
	return out.transpose();
}
