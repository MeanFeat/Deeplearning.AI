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
	params.W.push_back(MatrixXf::Random(A, B) * 0.5f);
	params.b.push_back(VectorXf::Zero(A, 1));
	cache.Z.push_back(MatrixXf::Zero(0, 0));
	cache.A.push_back(MatrixXf::Zero(0, 0));
	grads.dW.push_back(MatrixXf::Zero(0, 0));
	grads.db.push_back(MatrixXf::Zero(0, 0));
}

void Net::InitializeParameters(int inputSize, std::vector<int> hiddenSizes, int outputSize, vector<Activation> activations, float learningRate) {
	params.learningRate = learningRate;
	params.layerActivations = activations;
	params.layerSizes.push_back(inputSize);
	for(int l = 0; l < (int)hiddenSizes.size(); l++) {
		params.layerSizes.push_back(hiddenSizes[l]);
	}
	params.layerSizes.push_back(outputSize);
	AddLayer(hiddenSizes[0], inputSize);
	for(int h = 1; h < (int)hiddenSizes.size(); h++) {
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
	for(int i = 0; i < (int)params.layerSizes.size() - 1; i++) {
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
	return -(cache.A[cache.A.size() - 1].array().pow(2) - Y.array().pow(2)).sum()/Y.cols();
	//MatrixXf Output = cache.A[cache.A.size() - 1];
	//int m = (int)Y.cols();
	//float coeff = 1.0f / m;
	//return -((Y.cwiseProduct(Log(Output))) + (MatrixXf::Ones(1, m) - Y).cwiseProduct((Log(MatrixXf::Ones(1, m) - Output)))).sum() * coeff;
	
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
		default:
			break;
		}
		grads.dW[l] = coeff * (dZ * lowerA.transpose());
		grads.db[l] = coeff * dZ.rowwise().sum();
	}
}

void Net::UpdateParameters() {
	for(int i = 0; i < (int)grads.dW.size(); i++) {
		params.W[i] -= (params.learningRate * grads.mW[i]);
		params.b[i] -= (params.learningRate * grads.mb[i]);
	}
}

void Net::UpdateParametersWithMomentum() {
	for(int i = 0; i < (int)grads.dW.size(); i++) {
		grads.mW[i] = grads.dW[i] + grads.mW[i].normalized() * cache.cost * 0.025f;
		grads.mb[i] = grads.db[i] + grads.mb[i].normalized() * cache.cost * 0.025f;
		params.W[i] -= params.learningRate *grads.mW[i];
		params.b[i] -= params.learningRate *grads.mb[i];
	}
}

void Net::UpdateSingleStep(const MatrixXf X, const MatrixXf Y) {
	ForwardPropagation(X, true);
	cache.cost = ComputeCost(Y);
	BackwardPropagation(X, Y);
	UpdateParametersWithMomentum();
}

void Net::SaveNetwork() {
	
}

void Net::LoadNetwork() {
}