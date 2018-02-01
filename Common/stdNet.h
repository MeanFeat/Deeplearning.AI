#pragma once
#include <vector>
#include <iostream>
#include <fstream>
#include <Eigen/dense>

using namespace Eigen;
using namespace std;

inline MatrixXf Sigmoid(MatrixXf in) {
	return ((-1.0f*in).array().exp() + 1).cwiseInverse();
}

inline MatrixXf Tanh(MatrixXf in) {
	return (in.array().tanh());
}

inline MatrixXf Log(MatrixXf in) {
	return in.array().log();
}

struct NetGradients{
	MatrixXf dW1;
	VectorXf db1;
	MatrixXf dW2;
	VectorXf db2;
};

struct NetCache {
	MatrixXf Z1;
	MatrixXf A1;
	MatrixXf Z2;
	MatrixXf A2;
	float cost;
};

struct NetParameters {
	vector<int> layerSizes;
	MatrixXf W1;
	MatrixXf b1;
	MatrixXf W2;
	MatrixXf b2;
	float learningRate;
};

class Net {
public:
	Net();
	~Net();
	NetParameters GetParams();
	NetCache GetCache();
	void InitializeParameters(int inputSize, int hiddenSize, int outputSize);
	
	MatrixXf ForwardPropagation(MatrixXf X);
	MatrixXf GetHypothesis(MatrixXf input);
	float ComputeCost(MatrixXf A2, MatrixXf Y);
	void BackwardPropagation(MatrixXf X, MatrixXf Y);
	void UpdateParameters();
	void UpdateSingleStep(MatrixXf X, MatrixXf Y);

protected:
	NetParameters params;
	NetCache cache;
	NetGradients grads;
};