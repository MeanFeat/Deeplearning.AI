#pragma once
#include <vector>
#include <iostream>
#include <fstream>
#include <Eigen/dense>

using namespace Eigen;
using namespace std;

enum Activation {
	Linear,
	Sigmoid,
	Tanh,
	ReLU
};

inline MatrixXf CalcSigmoid( const MatrixXf &in) {
	return ((-1.0f*in).array().exp() + 1).cwiseInverse();
}

inline MatrixXf CalcTanh(const MatrixXf &in) {
	return (in.array().tanh());
}

inline MatrixXf CalcReLU(const MatrixXf &in) {
	return (in.cwiseMax(0.f));
}

inline MatrixXf Log(const MatrixXf &in) {
	return in.array().log();
}

struct NetParameters {
	vector<int> layerSizes;
	vector<Activation> layerActivations;
	vector<MatrixXf> W;
	vector<MatrixXf> b;
};

class Net {
public:
	Net();
	Net(int inputSize, std::vector<int> hiddenSizes, int outputSize, vector<Activation> activations);
	~Net();
	NetParameters GetParams();
	void SetParams(vector<MatrixXf> W, vector<MatrixXf> b);
	void AddLayer(int A, int B);
	static MatrixXf Activate(Activation act, const MatrixXf &In);	
	MatrixXf ForwardPropagation(const MatrixXf X);

	void SaveNetwork();
	void LoadNetwork();

protected:
	NetParameters params;
};