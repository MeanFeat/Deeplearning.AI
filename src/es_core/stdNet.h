#pragma once
#include <vector>
#include <iostream>
#include <fstream>
#include <Eigen/dense>
#include "types.h"

using namespace Eigen;
using namespace std;

inline MatrixXd CalcSigmoid( const MatrixXd &in) {
	return ((-1.0*in).array().exp() + 1).cwiseInverse();
}

inline MatrixXd CalcTanh(const MatrixXd &in) {
	return (in.array().tanh());
}

inline MatrixXd CalcReLU(const MatrixXd &in) {
	return (in.cwiseMax(0.f));
}

inline MatrixXd CalcLReLU(const MatrixXd &in) {
	return in.unaryExpr([](double elem) { return elem > 0.0 ? elem : elem * 0.01; });
}


inline MatrixXd Log(const MatrixXd &in) {
	return in.array().log();
}

struct NetParameters {
	vector<int> layerSizes;
	vector<Activation> layerActivations;
	vector<MatrixXd> W;
	vector<MatrixXd> b;
};

class Net {
public:
	Net();
	Net(int inputSize, std::vector<int> hiddenSizes, int outputSize, vector<Activation> activations);
	~Net();
	NetParameters &GetParams();
	void SetParams(vector<MatrixXd> W, vector<MatrixXd> b);
	void AddLayer(int A, int B);
	static MatrixXd Activate(Activation act, const MatrixXd &In);
	MatrixXd ForwardPropagation(const MatrixXd X);

	int Depth() {
		return (int)GetParams().layerSizes.size() - 1;
	}

	void SaveNetwork();
	void LoadNetwork();

protected:
	NetParameters params;
};