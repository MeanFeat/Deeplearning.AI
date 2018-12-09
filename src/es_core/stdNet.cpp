﻿#include "stdNet.h"

Net::Net(){
}

Net::Net(int inputSize, std::vector<int> hiddenSizes, int outputSize, vector<Activation> activations) {
	params.layerActivations = activations;
	params.layerSizes.push_back(inputSize);
	for(int l = 0; l < (int)hiddenSizes.size(); ++l) {
		params.layerSizes.push_back(hiddenSizes[l]);
	}
	params.layerSizes.push_back(outputSize);
	AddLayer(hiddenSizes[0], inputSize);
	for(int h = 1; h < (int)hiddenSizes.size(); ++h) {
		AddLayer(hiddenSizes[h], hiddenSizes[h - 1]);
	}
	AddLayer(outputSize, hiddenSizes.back());
}

Net::~Net() {
}

NetParameters &Net::GetParams() {
	return params;
}

void Net::SetParams(vector<MatrixXd> W, vector<MatrixXd> b) {
	params.W = W;
	params.b = b;
}

void Net::AddLayer(int A, int B) {
	params.W.push_back(MatrixXd::Random(A, B));
	params.b.push_back(MatrixXd::Zero(A, 1));
}

MatrixXd Net::Activate(Activation act, const MatrixXd &In) {
	switch(act) {
	case Linear:
		return In;
		break;
	case Sigmoid:
		return CalcSigmoid(In);
		break;
	case Tanh:
		return CalcTanh(In);
		break;
	case ReLU:
		return CalcReLU(In);
		break;
	case LReLU:
		return CalcLReLU(In);
		break;
	default:
		return In;
		break;
	}
}

MatrixXd Net::ForwardPropagation(const MatrixXd X) {
	MatrixXd lastOutput = X;
	for(int i = 0; i < (int)params.layerSizes.size() - 1; ++i) {
		lastOutput = Activate(params.layerActivations[i], (params.W[i] * lastOutput).colwise() + (VectorXd)params.b[i]);
	}
	return lastOutput;
}

void Net::SaveNetwork() {

}

void Net::LoadNetwork() {
}

