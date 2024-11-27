#pragma once

#include "stdNet.h"
#include "d_Matrix.h"
#include "d_math.h"
using namespace Eigen;
using namespace std;
struct d_NetBaseStructure {
	static void flushMat(std::vector<d_Matrix> &matrices)
	{
		for (d_Matrix &matrix : matrices)
		{
			matrix.free();
		}
		matrices.clear();
	}
};
struct d_NetTrainParameters : public d_NetBaseStructure {
    void clear()
	{
		flushMat(d_W);
		flushMat(d_b);
	}
	float coefficient;
	float learnCoeff;
	float learnMult;
	float learnRate;
	float regMod;
	float regMult;
	float regTerm;
	std::vector<d_Matrix> d_W;
	std::vector<d_Matrix> d_b;
	unsigned int trainExamplesCount;
};
struct d_NetTrainDerivatives : public d_NetBaseStructure{
	void clear()
	{
		flushMat(d_dW);
		flushMat(d_db);
	}
	std::vector<d_Matrix> d_dW;
	std::vector<d_Matrix> d_db;
};
struct d_NetCache  : public d_NetBaseStructure {
	void clear()
	{
		flushMat(d_A);
        flushMat(d_AT);
        flushMat(d_dZ);
	}
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
	void free();
	d_NetTrainer(stdNet *net, const Eigen::MatrixXf &data, const Eigen::MatrixXf &labels, float weightScale, float learnRate, float regTerm);
	~d_NetTrainer();
	//TODO: Make copy constructor
	static d_Matrix to_device(MatrixXf matrix);
	static MatrixXf to_host(d_Matrix d_matrix);
	d_NetTrainParameters &GetTrainParams();
	d_NetCache &GetCache();
	const d_NetProfiler *GetProfiler() const;
	stdNet *network;
	d_Matrix d_trainLabels;
	void BuildVisualization(const Eigen::MatrixXf &screen, int *buffer, int m, int k);
	void Visualization(int *buffer, int m, int k, bool discrete);
	void RefreshHostNetwork() const;
	void TrainSingleEpoch();
	d_Matrix Forward(const d_Matrix &Input) const;
	float CalcCost(const d_Matrix& Test, const d_Matrix& Labels) const;
	float GetCost()
	{
		return GetCache().cost;
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
	void ForwardTrain();
	void BackwardPropagation();
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
	void UpdateParameters();
	void UpdateParametersADAM();
};