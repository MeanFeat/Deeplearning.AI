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
	vector<d_MatrixXf> d_A;
	vector<d_MatrixXf> d_dZ;
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
	d_MatrixXf d_trainLabels;

	void AddLayer(int A, int B, float weightScale);

	MatrixXf Visualization(MatrixXf screen);
	
	d_MatrixXf ForwardTrain();
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
		MatrixXf* A = &cache.d_A[index].h_matrix();
		return (network->GetParams().W[index].transpose() * dZ).cwiseProduct(MatrixXf::Ones(A->rows(), A->cols()) - *A);
	}
	inline MatrixXf BackTanh(const MatrixXf dZ, int index) {
		MatrixXf* A = &cache.d_A[index].h_matrix();
		MatrixXf A1Squared = A->array().pow(2);
		return (network->GetParams().W[index].transpose() * dZ).cwiseProduct(MatrixXf::Ones(A->rows(), A->cols()) - (A1Squared));
	}
	inline MatrixXf BackReLU(const MatrixXf dZ, int index) {
		MatrixXf* A = &cache.d_A[index].h_matrix();
		return (network->GetParams().W[index].transpose() * dZ).cwiseProduct(A->unaryExpr([](float elem) { return elem > 0.f ? 1.f : 0.f; }));
	}

	inline MatrixXf BackLReLU(const MatrixXf dZ, int index) {
		MatrixXf* A = &cache.d_A[index].h_matrix();
		return (network->GetParams().W[index].transpose() * dZ).cwiseProduct(A->unaryExpr([](float elem) { return elem > 0.f ? 1.f : 0.01f; }));
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