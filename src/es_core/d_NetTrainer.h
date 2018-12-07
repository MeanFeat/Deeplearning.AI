#pragma once
#include <vector>
#include <iostream>
#include <fstream>
#include <Eigen/dense>
#include "stdNet.h"
#include "d_Matrix.h"
#include "d_Math.h"

using namespace Eigen;
using namespace std;

struct d_NetTrainParameters {
	vector<MatrixXf> dW;
	vector<MatrixXf> db;
	vector<d_MatrixXf> d_dW;
	vector<d_MatrixXf> d_db;
	vector<d_MatrixXf> d_W;
	vector<d_MatrixXf> d_b;
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

	void Visualization(MatrixXf screen, int * buffer, int m, int k, bool discrete);
	
	void ForwardTrain();
	float CalcCost(const MatrixXf h, const MatrixXf Y);
	void BackwardPropagation();
	void UpdateParameters();
	void UpdateParametersADAM();
	void UpdateSingleStep();

	inline void ModifyLearningRate(float m) {
		trainParams.learningRate = max(0.001f, trainParams.learningRate + m);
	}
	inline void ModifyRegTerm(float m) {
		trainParams.regTerm = max(FLT_EPSILON, trainParams.regTerm + m);
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


