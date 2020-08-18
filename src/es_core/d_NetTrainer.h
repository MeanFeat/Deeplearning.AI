#pragma once

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
	std::vector<d_Matrix> d_AT;
	std::vector<d_Matrix> d_dZ;
	float cost;
	float *d_cost;
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
	d_NetTrainer(Net *net, const Eigen::MatrixXf &data, const Eigen::MatrixXf &labels, float weightScale, float learnRate, float regTerm);
	~d_NetTrainer();
	d_NetTrainParameters GetTrainParams();
	d_NetCache GetCache();
	d_NetProfiler GetProfiler();
	Net *network;
	d_Matrix d_trainLabels;
	void BuildVisualization(const Eigen::MatrixXf &screen, int * buffer, int m, int k);
	void Visualization(int *buffer, int m, int k, bool discrete);
	void RefreshHostNetwork();
	void TrainSingleEpoch();
	float GetCost() {
		return GetCache().cost;
	}
	void SetCost(float c) {
		cache.cost = c;
	}
	float GetCoeff() {
		return trainParams.coefficiant;
	}
	float GetRegMultipier() {
		return trainParams.regMult;
	}
	inline void ModifyLearningRate(float m) {
		trainParams.learnCoeff = max(FLT_EPSILON, trainParams.learnCoeff + m);
	}
	inline void ModifyRegTerm(float m) {
		trainParams.regMod = max(FLT_EPSILON, trainParams.regMod + m);
	}
	unsigned int GetTrainExamplesCount() {
		return trainParams.trainExamplesCount;
	}
private:
	void CalcCost();
	d_NetCache cache;
	d_NetTrainParameters trainParams;
	d_NetTrainDerivatives derivative;
	d_NetTrainDerivatives momentum;
	d_NetTrainDerivatives momentumSqr;
	int *d_Buffer;
	std::vector<d_Matrix> d_VisualA;
	d_NetProfiler profiler;
	void AddLayer(int A, int B);
	void ForwardTrain();
	void BackwardPropagation();
	void UpdateParameters();
	void UpdateParametersADAM();
};