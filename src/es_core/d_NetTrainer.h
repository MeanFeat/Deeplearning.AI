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
	vector<d_Matrix> d_dW;
	vector<d_Matrix> d_db;
	vector<d_Matrix> d_W;
	vector<d_Matrix> d_b;
	double learningRate;
	double learningMod;
	double regTerm;
};

struct d_NetCache {
	vector<d_Matrix> d_A;
	vector<d_Matrix> d_dZ;
	double cost;
};

class d_NetTrainer {
public:
	d_NetTrainer();
	d_NetTrainer(Net *net, MatrixXd *data, MatrixXd *labels, double weightScale, double learnRate, double regTerm);
	~d_NetTrainer();
	
	d_NetTrainParameters GetTrainParams();
	d_NetCache GetCache();
	Net *network;
	MatrixXd *trainData;
	MatrixXd *trainLabels;
	d_Matrix d_trainLabels;

	void AddLayer(int A, int B);

	void Visualization(MatrixXd screen, int * buffer, int m, int k, bool discrete);
	
	void ForwardTrain();
	double CalcCost( MatrixXd h, MatrixXd Y);
	void BackwardPropagation();
	void UpdateParameters();
	void UpdateSingleParamADAM(MatrixXd * w, MatrixXd * d, MatrixXd * m, MatrixXd * mS);
	void UpdateParametersADAM();
	void UpdateSingleStep();

	inline void ModifyLearningRate(double m) {
		trainParams.learningRate = max(0.001, trainParams.learningRate + m);
	}
	inline void ModifyRegTerm(double m) {
		trainParams.regTerm = max(DBL_EPSILON, trainParams.regTerm + m);
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


