#pragma once
#include "types.h"
#include <Eigen/Dense>
inline Eigen::MatrixXf CalcSigmoid(const Eigen::MatrixXf &in) {
	//return in.unaryExpr([](float elem) { return 1.f / (1.f + exp(-elem)); });
	//return ((-1.f*in).array().exp() + 1).cwiseInverse();
	Eigen::MatrixXf out = Eigen::MatrixXf(in.rows(), in.cols());
	for (int i = 0; i < in.size(); i++) {
		*(out.data() + i) = 1.f / (1.f + expf(-(*(in.data() + i))));
	}
	return out;
}

inline Eigen::MatrixXf CalcTanh(const Eigen::MatrixXf &in) {
	return (in.array().tanh());
}

inline Eigen::MatrixXf CalcReLU(const Eigen::MatrixXf &in) {
	return (in.cwiseMax(0.f));
}

inline Eigen::MatrixXf CalcLReLU(const Eigen::MatrixXf &in) {
	return in.unaryExpr([](float elem) { return elem > 0.0f ? elem : elem * 0.01f; });
}

inline Eigen::MatrixXf CalcSine(const Eigen::MatrixXf &in) {
	return (in.array().sin());
}

inline Eigen::MatrixXf Log(const Eigen::MatrixXf &in) {
	return in.array().log();
}

struct stdNetParameters {
	std::vector<int> layerSizes;
	std::vector<Activation> layerActivations;
	std::vector<Eigen::MatrixXf> W;
	std::vector<Eigen::MatrixXf> b;
};

class stdNet {
public:
	stdNet();
	stdNet(int inputSize, const std::vector<int> &hiddenSizes, int outputSize, const std::vector<Activation> &activations);
	stdNet(const std::string &fName);
	~stdNet();
	void RandomInit(float scale);
	void SetParams(std::vector<Eigen::MatrixXf> W, std::vector<Eigen::MatrixXf> b);
	static Eigen::MatrixXf Activate(const Eigen::MatrixXf &In, Activation act);
	Eigen::MatrixXf ForwardPropagation(const Eigen::MatrixXf &X) const;
	int GetDepth();
	stdNetParameters &GetParams();
	float GetSumOfWeights();
	int GetNeuronCount();
	int GetInputSize() const;
	int GetOutputSize() const;
	int GetNodeCount();
	bool GetIsInitialized() const;
	std::string ToString();
	void SaveNetwork(const std::string& fName);
	void LoadNetwork(const std::string& fName);
private:
	void AddLayer(int a, int b);
	stdNetParameters params;
	bool IsInitialized = false;
};
