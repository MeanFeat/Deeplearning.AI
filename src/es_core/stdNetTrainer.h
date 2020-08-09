#pragma once
#include <vector>
#include <iostream>
#include <fstream>
#include <Eigen/dense>
#include "stdNet.h"

using namespace Eigen;
using namespace std;

struct NetTrainParameters {
	vector<MatrixXf> dW;
	vector<MatrixXf> db;
	float learningRate;
	float learningMod;
	float regTerm;
};

struct NetCache {
	vector<MatrixXf> Z;
	vector<MatrixXf> A;
	float cost;
};

class NetTrainer {
public:
	NetTrainer();
	NetTrainer(Net *net, MatrixXf *data, MatrixXf *labels, float weightScale, float learnRate, float regTerm);
	~NetTrainer();
	
	NetTrainParameters &GetTrainParams();
	NetCache &GetCache();
	Net *network;
	MatrixXf *trainData;
	MatrixXf *trainLabels;
	float coeff;
	void AddLayer(int A, int B);
	
	MatrixXf ForwardTrain();
	float CalcCost(const MatrixXf *h, const MatrixXf *Y);
	void BackwardPropagation();
	void BackLayer(MatrixXf &dZ, int l, const MatrixXf *LowerA);
	void UpdateParameters();
	void UpdateSingleParamADAM(MatrixXf * w, MatrixXf * d, MatrixXf * m, MatrixXf * mS);
	void UpdateParametersADAM();
	void BuildDropoutMask();
	void UpdateSingleStep();
	void ModifyLearningRate(float m);
	void ModifyRegTerm(float m);
	MatrixXf BackActivation(int l, const MatrixXf &dZ);
	MatrixXf BackSigmoid(const MatrixXf &dZ, int index);
	MatrixXf BackTanh(const MatrixXf &dZ, int index);
	MatrixXf BackReLU(const MatrixXf &dZ, int index);
	MatrixXf BackLReLU(const MatrixXf &dZ, int index);
	MatrixXf BackSine(const MatrixXf &dZ, int index);

protected:
	NetCache cache;
	NetParameters dropParams;
	NetTrainParameters trainParams;
	NetTrainParameters momentum;
	NetTrainParameters momentumSqr;
};