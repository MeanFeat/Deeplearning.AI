#include "stdNetTrainer.h"

NetTrainer::NetTrainer() {
}


NetTrainer::NetTrainer(Net *net, MatrixXd *data, MatrixXd *labels, double weightScale, double learnRate, double regTerm) {
	network = net;
	trainData = data;
	trainLabels = labels;
	int nodeCount = 0;
	for(int i = 0; i < (int)network->GetParams().layerSizes.size() - 1; ++i) {
		nodeCount += network->GetParams().layerSizes[i];
		network->GetParams().W[i] *= weightScale;
	}
	trainParams.learningMod = 1.0 / nodeCount;
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
	cache.Z.push_back(MatrixXd::Zero(0, 0));
	cache.A.push_back(MatrixXd::Zero(0, 0));
	trainParams.dW.push_back(MatrixXd::Zero(0, 0));
	trainParams.db.push_back(MatrixXd::Zero(0, 0));
	momentum.dW.push_back(MatrixXd::Zero(A, B));
	momentum.db.push_back(MatrixXd::Zero(A, 1));
	momentumSqr.dW.push_back(MatrixXd::Zero(A, B));
	momentumSqr.db.push_back(MatrixXd::Zero(A, 1));
}

MatrixXd NetTrainer::ForwardTrain() {
	MatrixXd lastOutput = *trainData;
	for(int i = 0; i < (int)network->GetParams().layerSizes.size() - 1; ++i) {
		cache.Z[i] = (network->GetParams().W[i] * lastOutput).colwise() + (VectorXd)network->GetParams().b[i];
		lastOutput = Net::Activate(network->GetParams().layerActivations[i], cache.Z[i]);
		cache.A[i] = lastOutput;
	}
	return lastOutput;
}

double NetTrainer::CalcCost(const MatrixXd h, MatrixXd Y) {
	double coeff = 1.f / Y.cols();
	double sumSqrW = 0.f;
	for(int w = 0; w < (int)network->GetParams().W.size() - 1; ++w) {
		sumSqrW += network->GetParams().W[w].array().pow(2).sum();
	}
	double regCost = 0.5 * double((trainParams.regTerm*trainParams.learningMod) * (sumSqrW / (2.0 * (double)trainLabels->cols())));
	return ((Y - h).array().pow(2).sum() * coeff) + regCost;
}

void NetTrainer::BackwardPropagation() {
	double m = (double)trainLabels->cols();
	double coeff = double(1.f / m);
	MatrixXd dZ = MatrixXd(cache.A.back() - * trainLabels);
	trainParams.dW.back() = coeff * (dZ * cache.A[cache.A.size() - 2].transpose());
	trainParams.db.back() = coeff * dZ.rowwise().sum();
	for(int l = (int)network->GetParams().layerActivations.size() - 2; l >= 0; --l) {
		MatrixXd lowerA = l > 0 ? cache.A[l - 1] : *trainData;
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
		trainParams.dW[l] = coeff * MatrixXd((dZ * lowerA.transpose()).array() + (0.5f * (trainParams.regTerm*trainParams.learningMod) * network->GetParams().W[l]).array());
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
#define BETA2 (1.f - DBL_EPSILON)
void NetTrainer::UpdateSingleParamADAM(MatrixXd *w, MatrixXd *d, MatrixXd *m, MatrixXd *mS) {
	*m = BETA1 * *m + (1 - BETA1) * *d;
	*mS = (BETA2 * *mS) + MatrixXd((1 - BETA2) * d->array().pow(2));
	*w -= (trainParams.learningRate*trainParams.learningMod)
		* MatrixXd((*m / (1 - pow(BETA1, 2))).array() / ((*mS / (1 - pow(BETA2, 2))).array().sqrt() + DBL_EPSILON));
}

void NetTrainer::UpdateParametersADAM() {
	for(int i = 0; i < (int)trainParams.dW.size(); ++i) {
		UpdateSingleParamADAM(&network->GetParams().W[i], &trainParams.dW[i], &momentum.dW[i], &momentumSqr.dW[i]);
		UpdateSingleParamADAM(&network->GetParams().b[i], &trainParams.db[i], &momentum.db[i], &momentumSqr.db[i]);
	}
}


void NetTrainer::BuildDropoutMask() {
	dropParams = dropParams;
	dropParams.W[0] = MatrixXd::Ones(dropParams.W[0].rows(), dropParams.W[0].cols());
	dropParams.b[0] = MatrixXd::Ones(dropParams.b[0].rows(), dropParams.b[0].cols());
	for(int i = 1; i < (int)dropParams.W.size() - 1; ++i) {
		for(int row = 0; row < dropParams.W[i].rows(); ++row) {
			double val = ((double)rand() / (RAND_MAX));
			if(val < 0.95) {
				dropParams.W[i].row(row) = MatrixXd::Ones(1, dropParams.W[i].cols());
				dropParams.b[i].row(row) = MatrixXd::Ones(1, dropParams.b[i].cols());
			} else {
				dropParams.W[i].row(row) = MatrixXd::Zero(1, dropParams.W[i].cols());
				dropParams.b[i].row(row) = MatrixXd::Zero(1, dropParams.b[i].cols());
			}
		}
	}
	dropParams.W[dropParams.W.size() - 1].row(0) = MatrixXd::Ones(1, dropParams.W[dropParams.W.size() - 1].cols());
	dropParams.b[dropParams.b.size() - 1].row(0) = MatrixXd::Ones(1, dropParams.b[dropParams.b.size() - 1].cols());
}

void NetTrainer::UpdateSingleStep() {
	//BuildDropoutMask();
	ForwardTrain();
	cache.cost = CalcCost(cache.A.back(), *trainLabels);
	BackwardPropagation();
	UpdateParametersADAM();
	cache.cost = CalcCost(cache.A.back(), *trainLabels);
}
