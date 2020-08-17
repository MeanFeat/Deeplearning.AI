#pragma once
#include "stdNet.h"

struct NetTrainParameters {
	std::vector<Eigen::MatrixXf> dW;
	std::vector<Eigen::MatrixXf> db;
	float learningRate;
	float learningMod;
	float regTerm;
};

struct NetCache {
	std::vector<Eigen::MatrixXf> Z;
	std::vector<Eigen::MatrixXf> A;
	float cost;
};

class NetTrainer {
public:
	NetTrainer();
	NetTrainer(Net *net, const Eigen::MatrixXf &data, const Eigen::MatrixXf &labels, float weightScale, float learnRate, float regTerm);
	~NetTrainer();

	Eigen::MatrixXf BackActivation(const Eigen::MatrixXf &dZ, int layerIndex);
	Eigen::MatrixXf BackLReLU(const Eigen::MatrixXf &dZ, int index);
	Eigen::MatrixXf BackReLU(const Eigen::MatrixXf &dZ, int index);
	Eigen::MatrixXf BackSigmoid(const Eigen::MatrixXf &dZ, int index);
	Eigen::MatrixXf BackSine(const Eigen::MatrixXf &dZ, int index);
	Eigen::MatrixXf BackTanh(const Eigen::MatrixXf &dZ, int index);
	Eigen::MatrixXf ForwardTrain();
	float CalcCost(const Eigen::MatrixXf &h, const Eigen::MatrixXf &Y);
	Net *network;
	NetCache &GetCache();
	NetTrainParameters &GetTrainParams();
	void AddLayer(int A, int B);
	void BackLayer(Eigen::MatrixXf &dZ, const Eigen::MatrixXf &LowerA, int layerIndex);
	void BackwardPropagation();
	void BuildDropoutMask();
	void ModifyLearningRate(float m);
	void ModifyRegTerm(float m);
	void UpdateParameters();
	void UpdateParametersADAM();
	void TrainSingleEpoch();

private:
	float coeff;
	Eigen::MatrixXf trainData;
	Eigen::MatrixXf trainLabels;
	NetCache cache;
	NetParameters dropParams;
	NetTrainParameters trainParams;
	NetTrainParameters momentum;
	NetTrainParameters momentumSqr;
};