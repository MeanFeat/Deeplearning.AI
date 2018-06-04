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
	return (in.cwiseMax(0));
}


inline MatrixXf Log(const MatrixXf &in) {
	return in.array().log();
}

struct NetGradients{
	vector<MatrixXf> dW;
	vector<MatrixXf> db;
};

struct NetCache {
	vector<MatrixXf> Z;
	vector<MatrixXf> A;
	float cost;
};

struct NetParameters {
	vector<int> layerSizes;
	vector<Activation> layerActivations;
	vector<MatrixXf> W;
	vector<MatrixXf> b;
	float learningRate;
};

class Net {
public:
	Net();
	~Net();
	NetParameters GetParams();
	NetCache GetCache();
	void AddLayer(int A, int B);
	void InitializeParameters(int inputSize, std::vector<int> hiddenLayers, int outputSize, vector<Activation> activations, float learningRate);

	MatrixXf Activate(Activation act, const MatrixXf &In);
	
	MatrixXf ForwardPropagation(const MatrixXf &X, bool training);
	float ComputeCost( const MatrixXf &Y);
	void BackwardPropagation(const MatrixXf &X, const MatrixXf &Y);
	void UpdateParameters();
	void UpdateSingleStep(const MatrixXf &X, const MatrixXf &Y);

	inline MatrixXf BackSigmoid(const MatrixXf &dZ, int index) {
		return (params.W[index + 1].transpose() * dZ).cwiseProduct(MatrixXf::Ones(cache.A[index].rows(), cache.A[index].cols()) - cache.A[index]);
	}

	inline MatrixXf BackTanh(const MatrixXf &dZ, int index) {
		MatrixXf A1Squared = cache.A[index].array().pow(2);
		return (params.W[index + 1].transpose() * dZ).cwiseProduct(MatrixXf::Ones(cache.A[index].rows(), cache.A[index].cols()) - (A1Squared));
	}

protected:
	NetParameters params;
	NetCache cache;
	NetGradients grads;
};