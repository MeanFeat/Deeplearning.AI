#pragma once
#include <vector>
#include <iostream>
#include <fstream>
#include <Eigen/dense>
#include "stdNet.h"

using namespace Eigen;
using namespace std;

struct NetTrainParameters {
	vector<MatrixXd> dW;
	vector<MatrixXd> db;
	double learningRate;
	double learningMod;
	double regTerm;
};

struct NetCache {
	vector<MatrixXd> Z;
	vector<MatrixXd> A;
	double cost;
};

class NetTrainer {
public:
	NetTrainer();
	NetTrainer(Net *net, MatrixXd *data, MatrixXd *labels, double weightScale, double learnRate, double regTerm);
	~NetTrainer();
	
	NetTrainParameters GetTrainParams();
	NetCache GetCache();
	Net *network;
	MatrixXd *trainData;
	MatrixXd *trainLabels;

	void AddLayer(int A, int B);
	
	MatrixXd ForwardTrain();
	double CalcCost(const MatrixXd h, const MatrixXd Y);
	void BackwardPropagation();
	void UpdateParameters();
	void UpdateParametersWithMomentum();
	void UpdateSingleParamADAM(MatrixXd * w, MatrixXd * d, MatrixXd * m, MatrixXd * mS);
	void UpdateParametersADAM();
	void BuildDropoutMask();
	void UpdateSingleStep();

	inline void ModifyLearningRate(double m) {
		trainParams.learningRate = max(0.001, trainParams.learningRate + m);
	}
	inline void ModifyRegTerm(double m) {
		trainParams.regTerm = max(DBL_EPSILON, trainParams.regTerm + m);
	}

	inline MatrixXd BackSigmoid(const MatrixXd dZ, int index) {
		return (network->GetParams().W[index + 1].transpose() * dZ).cwiseProduct(MatrixXd::Ones(cache.A[index].rows(), cache.A[index].cols()) - cache.A[index]);
	}
	inline MatrixXd BackTanh(const MatrixXd dZ, int index) {
		MatrixXd A1Squared = cache.A[index].array().pow(2);
		return (network->GetParams().W[index + 1].transpose() * dZ).cwiseProduct(MatrixXd::Ones(cache.A[index].rows(), cache.A[index].cols()) - (A1Squared));
	}
	inline MatrixXd BackReLU(const MatrixXd dZ, int index) {
		return (network->GetParams().W[index + 1].transpose() * dZ).cwiseProduct(cache.A[index].unaryExpr([](double elem) { return elem > 0.f ? 1.0 : 0.0; }));
	}

	inline MatrixXd BackLReLU(const MatrixXd dZ, int index) {
		return (network->GetParams().W[index + 1].transpose() * dZ).cwiseProduct(cache.A[index].unaryExpr([](double elem) { return elem > 0.f ? 1.0 : 0.01; }));
	}

protected:
	NetCache cache;
	NetParameters dropParams;
	NetTrainParameters trainParams;
	NetTrainParameters momentum;
	NetTrainParameters momentumSqr;
};