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

struct NetGradients {
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
	float learningMod;
	float regTerm;
};

class Net {
public:
	Net();
	~Net();
	NetParameters GetParams();
	NetCache GetCache();
	void AddLayer(int A, int B);
	void InitializeParameters(int inputSize, std::vector<int> hiddenLayers, int outputSize, vector<Activation> activations, float learningRate, float regTerm);

	MatrixXf Activate(Activation act, const MatrixXf &In);
	
	MatrixXf ForwardPropagation(const MatrixXf X, bool training = false);
	float ComputeCost(const MatrixXf h, const MatrixXf Y);
	void BackwardPropagation(const MatrixXf X, const MatrixXf Y);
	void UpdateParameters();
	void UpdateParametersWithMomentum();
	void UpdateParametersADAM();
	void BuildDropoutMask();
	void UpdateSingleStep(const MatrixXf X, const MatrixXf Y);

	inline void ModifyLearningRate(float m) {
		params.learningRate = max(0.001f, params.learningRate + m);
	}
	inline void ModifyRegTerm(float m) {
		params.regTerm = max(FLT_EPSILON, params.regTerm + m);
	}

	void SaveNetwork();
	void LoadNetwork();
	inline MatrixXf BackSigmoid(const MatrixXf dZ, int index) {
		return (params.W[index + 1].transpose() * dZ).cwiseProduct(MatrixXf::Ones(cache.A[index].rows(), cache.A[index].cols()) - cache.A[index]);
	}

	inline MatrixXf BackTanh(const MatrixXf dZ, int index) {
		MatrixXf A1Squared = cache.A[index].array().pow(2);
		return (params.W[index + 1].transpose() * dZ).cwiseProduct(MatrixXf::Ones(cache.A[index].rows(), cache.A[index].cols()) - (A1Squared));
	}
	inline MatrixXf BackReLU(const MatrixXf dZ, MatrixXf lowerA) {
		return (lowerA * dZ).unaryExpr([](float elem) { return elem > 0.f ? 1.f : 0.f; });
	}

protected:
	NetParameters params;
	NetParameters dropParams;
	NetCache cache;
	NetGradients grads;
	NetGradients momentum;
	NetGradients momentumSqr;
};