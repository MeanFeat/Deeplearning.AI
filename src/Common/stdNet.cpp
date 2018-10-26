#include "stdNet.h"

Net::Net() {
}
Net::~Net() {
}

NetParameters Net::GetParams() {
	return params;
}

NetCache Net::GetCache() {
	return cache;
}

void Net::AddLayer(int A, int B) {
	params.W.push_back(MatrixXf::Random(A, B) * 0.15f);
	params.b.push_back(VectorXf::Zero(A, 1));
	cache.Z.push_back(MatrixXf::Zero(0, 0));
	cache.A.push_back(MatrixXf::Zero(0, 0));
	grads.dW.push_back(MatrixXf::Zero(0, 0));
	grads.db.push_back(MatrixXf::Zero(0, 0));
	momentum.dW.push_back(MatrixXf::Zero(A, B));
	momentum.db.push_back(MatrixXf::Zero(A, 1));
	momentumSqr.dW.push_back(MatrixXf::Zero(A, B));
	momentumSqr.db.push_back(MatrixXf::Zero(A, 1));
}

void Net::InitializeParameters(int inputSize, std::vector<int> hiddenSizes,
							   int outputSize, vector<Activation> activations,
							   float learningRate, float regTerm) {
	params.learningRate = learningRate;
	params.regTerm = regTerm;
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
	return;
}

MatrixXf Net::Activate( Activation act, const MatrixXf &In) {
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
	default:
		return In;
		break;
	}
}

MatrixXf Net::ForwardPropagation(const MatrixXf X, bool training) {
	MatrixXf lastOutput = X;
	for(int i = 0; i < (int)params.layerSizes.size() - 1; ++i) {
		MatrixXf Z = (params.W[i] * lastOutput).colwise() + (VectorXf)params.b[i];
		lastOutput = Activate(params.layerActivations[i],Z);
		if(training) {
			cache.Z[i] = Z;
			cache.A[i] = lastOutput;
		}
	}
	return lastOutput;
}

float Net::ComputeCost(const MatrixXf Y) {
	float sumSqrW = 0.f;
	for(int w = 0; w < (int)params.W.size() - 1; ++w) {
		sumSqrW += params.W[w].array().pow(2).sum();
	}
	float regCost = float(params.regTerm * (sumSqrW / (2.f * (float)Y.cols())));
	return -(((cache.A[cache.A.size() - 1].array().pow(2) - Y.array().pow(2)).sum()/Y.cols())) + regCost;	
}

void Net::BackwardPropagation(const MatrixXf X, const MatrixXf Y) {
	int m = (int)Y.cols();
	float coeff = float(1.f / m);
	MatrixXf dZ = cache.A.back() - Y;
	grads.dW.back() = coeff * (dZ * cache.A[cache.A.size()-2].transpose());
	grads.db.back() = coeff * dZ.rowwise().sum();
	for(int l = params.layerActivations.size() - 2; l >= 0; --l) {
		MatrixXf lowerA = l > 0 ? cache.A[l-1] : X;
		switch(params.layerActivations[l]) {
		case Sigmoid:
			dZ = BackSigmoid(dZ, l);
			break;
		case Tanh:			
			dZ = BackTanh(dZ, l);
			break;
		case ReLU:
			dZ = BackReLU(dZ, lowerA);
			grads.dW[l] = coeff * (dZ * lowerA.transpose());
			grads.db[l] = coeff * dZ.rowwise().sum();
			break;
		default:
			break;
		}
		grads.dW[l] = coeff * MatrixXf((dZ * lowerA.transpose()).array() + (params.regTerm * params.W[l]).array());
		grads.db[l] = coeff * dZ.rowwise().sum();
	}
	
}

void Net::UpdateParameters() {
	for(int i = 0; i < (int)grads.dW.size(); ++i) {
		params.W[i] -= (params.learningRate * grads.dW[i]);
		params.b[i] -= (params.learningRate * grads.db[i]);
	}
}

void Net::UpdateParametersWithMomentum() {
	for(int i = 0; i < (int)grads.dW.size(); ++i) {
		momentum.dW[i] = grads.dW[i] + momentum.dW[i].normalized() * cache.cost * 0.025;
		momentum.db[i] = grads.db[i] + momentum.db[i].normalized() * cache.cost * 0.025;
		params.W[i] -= params.learningRate * momentum.dW[i];
		params.b[i] -= params.learningRate * momentum.db[i];
	}
}

#define BETA1 0.9
#define BETA2 (1.f - FLT_EPSILON)
void Net::UpdateParametersADAM() {
	for(int i = 0; i < (int)grads.dW.size(); ++i) {
		NetGradients vCorrected = momentum;
		NetGradients sCorrected = momentumSqr;
		momentum.dW[i] = BETA1 * momentum.dW[i] + (1 - BETA1) * grads.dW[i];
		momentum.db[i] = BETA1 * momentum.db[i] + (1 - BETA1) * grads.db[i];
		vCorrected.dW[i] = momentum.dW[i] / (1 - pow(BETA1, 2));
		vCorrected.db[i] = momentum.db[i] / (1 - pow(BETA1, 2));
		momentumSqr.dW[i] = (BETA2 * momentumSqr.dW[i]) + ((1 - BETA2) * MatrixXf(grads.dW[i].array().pow(2)));
		momentumSqr.db[i] = (BETA2 * momentumSqr.db[i]) + ((1 - BETA2) * MatrixXf(grads.db[i].array().pow(2)));
		sCorrected.dW[i] = momentumSqr.dW[i] / (1 - pow(BETA2, 2));
		sCorrected.db[i] = momentumSqr.db[i] / (1 - pow(BETA2, 2));
		params.W[i] -= params.learningRate * MatrixXf(vCorrected.dW[i].array() / (sCorrected.dW[i].array().sqrt() + FLT_EPSILON));
		params.b[i] -= params.learningRate * MatrixXf(vCorrected.db[i].array() / (sCorrected.db[i].array().sqrt() + FLT_EPSILON));
	}
}


void Net::BuildDropoutMask() {
	dropParams = params;
	dropParams.W[0] = MatrixXf::Ones(dropParams.W[0].rows(), dropParams.W[0].cols());
	dropParams.b[0] = MatrixXf::Ones(dropParams.b[0].rows(), dropParams.b[0].cols());
	for(int i = 1; i < (int)dropParams.W.size() - 1; ++i) {
		for(int row = 0; row < dropParams.W[i].rows(); ++row) {
			float val = ((float)rand() / (RAND_MAX));
			if(val < 0.95) {
				dropParams.W[i].row(row) = MatrixXf::Ones(1, dropParams.W[i].cols());
				dropParams.b[i].row(row) = MatrixXf::Ones(1, dropParams.b[i].cols());
			} else {
				dropParams.W[i].row(row) = MatrixXf::Zero(1, dropParams.W[i].cols());
				dropParams.b[i].row(row) = MatrixXf::Zero(1, dropParams.b[i].cols());
			}
		}		
	}
	dropParams.W[dropParams.W.size() - 1].row(0) = MatrixXf::Ones(1, dropParams.W[dropParams.W.size() - 1].cols());
	dropParams.b[dropParams.b.size() - 1].row(0) = MatrixXf::Ones(1, dropParams.b[dropParams.b.size() - 1].cols());
}

void Net::UpdateSingleStep(const MatrixXf X, const MatrixXf Y) {
	//BuildDropoutMask();
	ForwardPropagation(X, true);
	cache.cost = ComputeCost(Y);
	BackwardPropagation(X, Y);
	UpdateParametersADAM();
}

void Net::SaveNetwork() {
	
}

void Net::LoadNetwork() {
}
