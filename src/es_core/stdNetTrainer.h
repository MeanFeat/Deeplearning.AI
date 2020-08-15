#pragma once
#ifndef STD_NET_TRAINER_H
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
	NetTrainer(Net &net, MatrixXf &data, MatrixXf &labels, float weightScale, float learnRate, float regTerm);
	~NetTrainer();
	NetTrainParameters &GetTrainParams();
	NetCache &GetCache();
	void AddLayer(int A, int B, float weightScale);
	MatrixXf ForwardTrain();
	float CalcCost(const MatrixXf & h, const MatrixXf & Y);
	void BackwardPropagation();
	void BackLayer(MatrixXf &dZ, const MatrixXf *LowerA, int layerIndex);
	void UpdateParameters();
	void UpdateParametersWithMomentum();
	void UpdateParametersADAM();
	void BuildDropoutMask();
	void UpdateSingleStep();
	void ModifyLearningRate(float m);
	void ModifyRegTerm(float m);
	MatrixXf BackActivation(const MatrixXf &dZ, int layerIndex);
	MatrixXf BackSigmoid(const MatrixXf &dZ, int index);
	MatrixXf BackTanh(const MatrixXf &dZ, int index);
	MatrixXf BackReLU(const MatrixXf &dZ, int index);
	MatrixXf BackLReLU(const MatrixXf &dZ, int index);
	MatrixXf BackSine(const MatrixXf &dZ, int index);

private:
	Net *network;
	MatrixXf *trainData;
	MatrixXf *trainLabels;
	float coeff;
	NetCache cache;
	NetParameters dropParams;
	NetTrainParameters trainParams;
	NetTrainParameters momentum;
	NetTrainParameters momentumSqr;
};
#define STD_NET_TRAINER_H
#endif