#pragma once
#include <vector>
#include <iostream>
#include <fstream>
#include <Eigen/dense>
#include "types.h"

using namespace Eigen;
using namespace std;

inline MatrixXf CalcSigmoid( const MatrixXf &in) {
	//return in.unaryExpr([](float elem) { return 1.f / (1.f + exp(-elem)); });
	//return ((-1.f*in).array().exp() + 1).cwiseInverse();
	MatrixXf out = MatrixXf(in.rows(),in.cols());
	for (int i = 0; i < in.size(); i++)	{
		*(out.data() + i) = 1.f / (1.f + expf(-(*(in.data() + i))));
	}
	return out;
}

inline MatrixXf CalcTanh(const MatrixXf &in) {
	return (in.array().tanh());
}

inline MatrixXf CalcReLU(const MatrixXf &in) {
	return (in.cwiseMax(0.f));
}

inline MatrixXf CalcLReLU(const MatrixXf &in) {
	return in.unaryExpr([](float elem) { return elem > 0.0f ? elem : elem * 0.01f; });
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
	NetParameters &GetParams();
	void SetParams(vector<MatrixXf> W, vector<MatrixXf> b);
	void AddLayer(int A, int B);
	static MatrixXf Activate(Activation act, const MatrixXf &In);
	MatrixXf ForwardPropagation(const MatrixXf X);

	int Depth() {
		return (int)GetParams().layerSizes.size() - 1;
	}

	void SaveNetwork();
	void LoadNetwork();

protected:
	NetParameters params;
};