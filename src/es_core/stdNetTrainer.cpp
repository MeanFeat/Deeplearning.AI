#include "stdNetTrainer.h"

NetTrainer::NetTrainer() {
}


NetTrainer::NetTrainer(Net *net, MatrixXf *data, MatrixXf *labels, float weightScale, float learnRate, float regTerm) {
	network = net;
	trainData = data;
	trainLabels = labels;
	int nodeCount = 0;
	for(int i = 0; i < (int)network->GetParams().layerSizes.size() - 1; ++i) {
		nodeCount += network->GetParams().layerSizes[i];
		network->GetParams().W[i] *= weightScale;
	}
	trainParams.learningMod = 1.f / nodeCount;
	trainParams.learningRate = learnRate;
	trainParams.regTerm = regTerm;
	for(int i = 1; i < (int)network->GetParams().layerSizes.size(); ++i) {
		AddLayer((int)network->GetParams().layerSizes[i], (int)network->GetParams().layerSizes[i - 1]);
	}
}

NetTrainer::~NetTrainer() {
}


NetTrainParameters NetTrainer::GetTrainParams() {
	return trainParams;
}

NetCache NetTrainer::GetCache() {
	return cache;
}

void NetTrainer::AddLayer(int A, int B) {
	cache.Z.push_back(MatrixXf::Zero(0, 0));
	cache.A.push_back(MatrixXf::Zero(0, 0));
	trainParams.dW.push_back(MatrixXf::Zero(0, 0));
	trainParams.db.push_back(MatrixXf::Zero(0, 0));
	momentum.dW.push_back(MatrixXf::Zero(A, B));
	momentum.db.push_back(MatrixXf::Zero(A, 1));
	momentumSqr.dW.push_back(MatrixXf::Zero(A, B));
	momentumSqr.db.push_back(MatrixXf::Zero(A, 1));
}

MatrixXf NetTrainer::ForwardTrain() {
	MatrixXf lastOutput = *trainData;
	for(int i = 0; i < (int)network->GetParams().layerSizes.size() - 1; ++i) {
		cache.Z[i] = (network->GetParams().W[i] * lastOutput).colwise() + (VectorXf)network->GetParams().b[i];
		lastOutput = Net::Activate(network->GetParams().layerActivations[i], cache.Z[i]);
		cache.A[i] = lastOutput;
	}
	return lastOutput;
}

float NetTrainer::CalcCost(const MatrixXf h, MatrixXf Y) {
	float coeff = 1.f / Y.cols();
	float sumSqrW = 0.f;
	for(int w = 0; w < (int)network->GetParams().W.size() - 1; ++w) {
		sumSqrW += network->GetParams().W[w].array().pow(2).sum();
	}
	float regCost = 0.5f * float((trainParams.regTerm*trainParams.learningMod) * (sumSqrW / (2.f * (float)trainLabels->cols())));
	return ((Y - h).array().pow(2).sum() * coeff) + regCost;
}

void NetTrainer::BackwardPropagation() {
	float m = (float)trainLabels->cols();
	float coeff = float(1.f / m);
	MatrixXf dZ = MatrixXf(cache.A.back() - * trainLabels);
	trainParams.dW.back() = coeff * (dZ * cache.A[cache.A.size() - 2].transpose());
	trainParams.db.back() = coeff * dZ.rowwise().sum();
	for(int l = (int)network->GetParams().layerActivations.size() - 2; l >= 0; --l) {
		MatrixXf lowerA = l > 0 ? cache.A[l - 1] : *trainData;
		switch(network->GetParams().layerActivations[l]) {
		case Sigmoid:
			dZ = BackSigmoid(dZ, l);
			break;
		case Tanh:
			dZ = BackTanh(dZ, l);
			break;
		case ReLU:
			dZ = BackReLU(dZ, l);
			break;
		case LReLU:
			dZ = BackLReLU(dZ, l);
			break;
		default:
			break;
		}
		trainParams.dW[l] = coeff * MatrixXf((dZ * lowerA.transpose()).array() + (0.5f * (trainParams.regTerm*trainParams.learningMod) * network->GetParams().W[l]).array());
		trainParams.db[l] = coeff * dZ.rowwise().sum();
	}
}

void NetTrainer::UpdateParameters() {
	for(int i = 0; i < (int)trainParams.dW.size(); ++i) {
		network->GetParams().W[i] -= ((trainParams.learningRate*trainParams.learningMod) * trainParams.dW[i]);
		network->GetParams().b[i] -= ((trainParams.learningRate*trainParams.learningMod) * trainParams.db[i]);
	}
}

void NetTrainer::UpdateParametersWithMomentum() {
	for(int i = 0; i < (int)trainParams.dW.size(); ++i) {
		momentum.dW[i] = trainParams.dW[i] + momentum.dW[i].normalized() * cache.cost * 0.025;
		momentum.db[i] = trainParams.db[i] + momentum.db[i].normalized() * cache.cost * 0.025;
		network->GetParams().W[i] -= (trainParams.learningRate*trainParams.learningMod) * momentum.dW[i];
		network->GetParams().b[i] -= (trainParams.learningRate*trainParams.learningMod) * momentum.db[i];
	}
}

#define BETA1 0.9
#define BETA2 (1.f - FLT_EPSILON)
void NetTrainer::UpdateSingleParamADAM(MatrixXf *w, MatrixXf *d, MatrixXf *m, MatrixXf *mS) {
	*m = BETA1 * *m + (1 - BETA1) * *d;
	*mS = (BETA2 * *mS) + MatrixXf((1 - BETA2) * d->array().pow(2));
	*w -= (trainParams.learningRate*trainParams.learningMod)
		* MatrixXf((*m / (1 - pow(BETA1, 2))).array() / ((*mS / (1 - pow(BETA2, 2))).array().sqrt() + FLT_EPSILON));
}

void NetTrainer::UpdateParametersADAM() {
	for(int i = 0; i < (int)trainParams.dW.size(); ++i) {
		UpdateSingleParamADAM(&network->GetParams().W[i], &trainParams.dW[i], &momentum.dW[i], &momentumSqr.dW[i]);
		UpdateSingleParamADAM(&network->GetParams().b[i], &trainParams.db[i], &momentum.db[i], &momentumSqr.db[i]);
	}
}


void NetTrainer::BuildDropoutMask() {
	dropParams = dropParams;
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

void NetTrainer::UpdateSingleStep() {
	//BuildDropoutMask();
	ForwardTrain();
	BackwardPropagation();
	UpdateParametersADAM();
	cache.cost = CalcCost(cache.A.back(), *trainLabels);
}
