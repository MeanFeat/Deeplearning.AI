#pragma once
#include <vector>
#include <iostream>
#include <fstream>
#include "stdNet.h"
#include "d_Matrix.h"
#include "d_Math.h"

struct d_NetTrainParameters {
	float coefficiant;
	float learnCoeff;
	float learnMult;
	float learnRate;
	float regMod;
	float regMult;
	float regTerm;
	std::vector<d_Matrix> d_b;
	std::vector<d_Matrix> d_W;
	unsigned int trainExamplesCount;
};
struct d_NetTrainDerivatives {
	std::vector<d_Matrix> d_dW;
	std::vector<d_Matrix> d_db;
};
struct d_NetCache {
	std::vector<d_Matrix> d_A;
	std::vector<d_Matrix> d_dZ;
	float cost;
};
struct d_NetProfiler {
	float backpropTime;
	float calcCostTime;
	float forwardTime;
	float updateTime;
	float visualizationTime;
};
class d_NetTrainer {
public:
	d_NetTrainer();
	d_NetTrainer(Net *net, Eigen::MatrixXf *data, Eigen::MatrixXf *labels, float weightScale, float learnRate, float regTerm);
	~d_NetTrainer();
	d_NetTrainParameters GetTrainParams();
	d_NetCache GetCache();
	d_NetProfiler GetProfiler();
	Net *network;
	d_Matrix d_trainLabels;
	void BuildVisualization(Eigen::MatrixXf screen, int * buffer, int m, int k);
	void Visualization(int * buffer, int m, int k, bool discrete);
	void UpdateHostNetwork();
	void UpdateSingleStep();
	float CalcCost();
	float GetCost() {
		return GetCache().cost;
	}
	float Coeff() {
		return trainParams.coefficiant;
	}
	float RegMultipier() {
		return trainParams.regMult;
	}
	inline void ModifyLearningRate(float m) {
		trainParams.learnCoeff = max(FLT_EPSILON, trainParams.learnCoeff + m);
	}
	inline void ModifyRegTerm(float m) {
		trainParams.regMod = max(FLT_EPSILON, trainParams.regMod + m);
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
	std::vector<d_Matrix> d_VisualA;
	d_NetProfiler profiler;
private:
	void AddLayer(int A, int B);
	void ForwardTrain();
	void BackwardPropagation();
	void UpdateParameters();
	void UpdateParametersADAM();
};