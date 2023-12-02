#pragma once

#include "stdNet.h"
#include "d_Matrix.h"
#include "d_Math.h"

struct d_NetTrainParameters {
	float coefficient;
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
	d_NetProfiler GetProfiler() const;
	Net *network;
	d_Matrix d_trainLabels;
	void BuildVisualization(const Eigen::MatrixXf &screen, int *buffer, int m, int k);
	void Visualization(int *buffer, int m, int k, bool discrete);
	void RefreshHostNetwork() const;
	void TrainSingleEpoch();
	float GetCost() {
		return GetCache().cost;
	}
	void SetCost(const float c) {
		cache.cost = c;
	}
	float GetCoeff() const {
		return trainParams.coefficient;
	}
	float GetRegMultiplier() const {
		return trainParams.regMult;
	}
	inline void ModifyLearningRate(const float m) {
		trainParams.learnMult = max(FLT_EPSILON, trainParams.learnMult + (m * trainParams.learnCoeff));
	}
	inline void ModifyRegTerm(const float m) {
		trainParams.regMod = max(FLT_EPSILON, trainParams.regMod + (m * trainParams.learnCoeff));
	}
	unsigned int GetTrainExamplesCount() const {
		return trainParams.trainExamplesCount;
	}
	d_NetTrainDerivatives GetDerivatives() {
		return derivative;
	}
	void BackwardPropagation();
	void CalcCost();
	void ForwardTrain();
private:
	d_NetCache cache;
	d_NetTrainParameters trainParams;
	d_NetTrainDerivatives derivative;
	d_NetTrainDerivatives momentum;
	d_NetTrainDerivatives momentumSqr;
	int *d_Buffer;
	std::vector<d_Matrix> d_VisualA;
	d_NetProfiler profiler;
	void AddLayer(int A, int B);
	void UpdateParameters();
	void UpdateParametersADAM();
};