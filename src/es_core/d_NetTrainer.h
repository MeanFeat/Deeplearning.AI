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
	vector<d_Matrix> d_W;
	vector<d_Matrix> d_b;
	double learnRate;
	double learnCoeff;
	double learnMult;
	double regTerm;
	double regMod;
	double regMult;
	double coefficiant;
	unsigned int trainExamplesCount;
};

struct d_NetTrainDerivatives {
	vector<d_Matrix> d_dW;
	vector<d_Matrix> d_db;
};

struct d_NetCache {
	vector<d_Matrix> d_A;
	vector<d_Matrix> d_dZ;
	double cost;
};

struct d_NetProfiler {
	float forwardTime;
	float backpropTime;
	float updateTime;
	float calcCostTime;
};

class d_NetTrainer {
public:
	d_NetTrainer();
	d_NetTrainer(Net *net, MatrixXd *data, MatrixXd *labels, double weightScale, double learnRate, double regTerm);
	~d_NetTrainer();
	
	d_NetTrainParameters GetTrainParams();
	d_NetCache GetCache();
	d_NetProfiler GetProfiler();
	Net *network;
	d_Matrix d_trainLabels;

	void BuildVisualization(MatrixXd screen, int * buffer, int m, int k);
	void Visualization(int * buffer, int m, int k, bool discrete);
	void UpdateHostNetwork();
	void UpdateSingleStep();
	double CalcCost();

	double GetCost() {
		return GetCache().cost;
	}

	double Coeff() {
		return trainParams.coefficiant;
	}
	double RegMultipier() {
		return trainParams.regMult;
	}

	inline void ModifyLearningRate(double m) {
		trainParams.learnCoeff = max(DBL_EPSILON, trainParams.learnCoeff + m);
	}
	inline void ModifyRegTerm(double m) {
		trainParams.regMod = max(DBL_EPSILON, trainParams.regMod + m);
	}

	unsigned int TrainExamplesCount() {
		return trainParams.trainExamplesCount;
	}
	
protected:
	d_NetCache cache;
	d_NetTrainParameters trainParams;
	d_NetTrainDerivatives derivative;
	d_NetTrainDerivatives momentum;
	d_NetTrainDerivatives momentumSqr;
	int *d_Buffer;
	vector<d_Matrix> d_VisualA;
	d_NetProfiler profiler;

private:
	void AddLayer(int A, int B);
	void ForwardTrain();
	void BackwardPropagation();
	void UpdateParameters();
	void UpdateParametersADAM();
};


