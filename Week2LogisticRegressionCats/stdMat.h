#pragma once
#include <Eigen/dense>
#include <vector>

using namespace Eigen;
using namespace std;

inline MatrixXf Sigmoid(MatrixXf in) {
	return ((-1.0*in).array().exp() + 1 ).cwiseInverse();
}

inline MatrixXf Log(MatrixXf in) { 
	return in.array().log();
}