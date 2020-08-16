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
	NetTrainer(Net *net, Eigen::MatrixXf *data, Eigen::MatrixXf *labels, float weightScale, float learnRate, float regTerm);
	~NetTrainer();

	NetTrainParameters &GetTrainParams();
	NetCache &GetCache();
	Net *network;
	Eigen::MatrixXf *trainData;
	Eigen::MatrixXf *trainLabels;
	float coeff;
	void AddLayer(int A, int B);

	Eigen::MatrixXf ForwardTrain();
	float CalcCost(const Eigen::MatrixXf *h, const Eigen::MatrixXf *Y);
	void BackwardPropagation();
	void BackLayer(Eigen::MatrixXf &dZ, int l, const Eigen::MatrixXf *LowerA);
	void UpdateParameters();
	void UpdateParametersADAM();
	void BuildDropoutMask();
	void UpdateSingleStep();
	void ModifyLearningRate(float m);
	void ModifyRegTerm(float m);
	Eigen::MatrixXf BackActivation(int l, const Eigen::MatrixXf &dZ);
	Eigen::MatrixXf BackSigmoid(const Eigen::MatrixXf &dZ, int index);
	Eigen::MatrixXf BackTanh(const Eigen::MatrixXf &dZ, int index);
	Eigen::MatrixXf BackReLU(const Eigen::MatrixXf &dZ, int index);
	Eigen::MatrixXf BackLReLU(const Eigen::MatrixXf &dZ, int index);
	Eigen::MatrixXf BackSine(const Eigen::MatrixXf &dZ, int index);

protected:
	NetCache cache;
	NetParameters dropParams;
	NetTrainParameters trainParams;
	NetTrainParameters momentum;
	NetTrainParameters momentumSqr;
};