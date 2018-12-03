#pragma once
#include <vector>
#include <iostream>
#include <fstream>
#include <Eigen/dense>
#include "stdNet.h"
#include "d_Matrix.h"

using namespace Eigen;
using namespace std;

struct d_NetTrainParameters {
	vector<MatrixXf> dW;
	vector<MatrixXf> db;
	float learningRate;
	float learningMod;
	float regTerm;
};

struct d_NetCache {
	vector<MatrixXf> Z;
	vector<MatrixXf> A;
	float cost;
};

class d_NetTrainer {
public:
	d_NetTrainer();
	d_NetTrainer(Net *net, MatrixXf *data, MatrixXf *labels, float weightScale, float learnRate, float regTerm);
	~d_NetTrainer();
	
	d_NetTrainParameters GetTrainParams();
	d_NetCache GetCache();
	Net *network;
	MatrixXf *trainData;
	MatrixXf *trainLabels;

	void AddLayer(int A, int B, float weightScale);
	
	MatrixXf ForwardTrain();
	float CalcCost(const MatrixXf h, const MatrixXf Y);
	void BackwardPropagation();
	void UpdateParameters();
	void CleanUpAll();
	void UpdateParametersADAM();
	void UpdateSingleStep();


	inline void ModifyLearningRate(float m) {
		trainParams.learningRate = max(0.001f, trainParams.learningRate + m);
	}
	inline void ModifyRegTerm(float m) {
		trainParams.regTerm = max(FLT_EPSILON, trainParams.regTerm + m);
	}

	inline MatrixXf BackSigmoid(const MatrixXf dZ, int index) {
		return (network->GetParams().W[index].transpose() * dZ).cwiseProduct(MatrixXf::Ones(cache.A[index].rows(), cache.A[index].cols()) - cache.A[index]);
	}
	inline MatrixXf BackTanh(const MatrixXf dZ, int index) {
		MatrixXf A1Squared = cache.A[index].array().pow(2);
		return (network->GetParams().W[index].transpose() * dZ).cwiseProduct(MatrixXf::Ones(cache.A[index].rows(), cache.A[index].cols()) - (A1Squared));
	}
	inline MatrixXf BackReLU(const MatrixXf dZ, int index) {
		return (network->GetParams().W[index].transpose() * dZ).cwiseProduct(cache.A[index].unaryExpr([](float elem) { return elem > 0.f ? 1.f : 0.f; }));
	}

	inline MatrixXf BackLReLU(const MatrixXf dZ, int index) {
		return (network->GetParams().W[index].transpose() * dZ).cwiseProduct(cache.A[index].unaryExpr([](float elem) { return elem > 0.f ? 1.f : 0.01f; }));
	}

	unsigned int GetExamplesCount() {
		return trainExamplesCount;
	}
	
protected:
	d_NetCache cache;
	d_NetTrainParameters trainParams;
	d_NetTrainParameters momentum;
	d_NetTrainParameters momentumSqr;

	unsigned int trainExamplesCount;
};