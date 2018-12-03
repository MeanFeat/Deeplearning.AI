#include "d_NetTrainer.h"

d_NetTrainer::d_NetTrainer() {
}


d_NetTrainer::d_NetTrainer(Net *net, MatrixXf *data, MatrixXf *labels, float weightScale, float learnRate, float regTerm) {
	network = net;
	trainData = data;
	trainLabels = labels;
	trainExamplesCount = (unsigned int)data->cols();
	int nodeCount = 0;
	for(int i = 0; i < (int)network->GetParams().layerSizes.size() - 1; ++i) {
		nodeCount += network->GetParams().layerSizes[i];
		network->GetParams().W[i] *= weightScale;
	}
	trainParams.learningMod = 1.f / nodeCount;
	trainParams.learningRate = learnRate;
	trainParams.regTerm = regTerm;
	cache.A.push_back(*trainData);
	for(int h = 1; h < (int)network->GetParams().layerSizes.size(); ++h) {
		AddLayer((int)network->GetParams().layerSizes[h], (int)network->GetParams().layerSizes[h - 1], weightScale);
	}
}

d_NetTrainer::~d_NetTrainer() {
}


d_NetTrainParameters d_NetTrainer::GetTrainParams() {
	return trainParams;
}

d_NetCache d_NetTrainer::GetCache() {
	return cache;
}

void d_NetTrainer::AddLayer(int A, int B, float weightScale) {
	cache.Z.push_back(MatrixXf::Zero(A, trainExamplesCount));
	cache.A.push_back(MatrixXf::Zero(A, trainExamplesCount));
	trainParams.dW.push_back(MatrixXf::Zero(A, B));
	trainParams.db.push_back(MatrixXf::Zero(A, 1));
	momentum.dW.push_back(MatrixXf::Zero(A, B));
	momentum.db.push_back(MatrixXf::Zero(A, 1));
	momentumSqr.dW.push_back(MatrixXf::Zero(A, B));
	momentumSqr.db.push_back(MatrixXf::Zero(A, 1));
}

MatrixXf d_NetTrainer::ForwardTrain() {	
	MatrixXf lastOutput = *trainData;
	for(int i = 0; i < (int)network->GetParams().layerSizes.size() - 1; ++i) {
		d_MatrixXf d_lastOutput = d_MatrixXf(&cache.A[i]);
		d_MatrixXf d_W = d_MatrixXf(&network->GetParams().W[i]);
		d_MatrixXf d_b = d_MatrixXf(&network->GetParams().b[i]);
		MatrixXf tempA = MatrixXf::Zero(d_W.rows(), d_lastOutput.cols());
		d_MatrixXf d_A = d_MatrixXf(&tempA);
		d_forwardLayer(d_A.d_data(), d_W.d_data(), d_lastOutput.d_data(), d_b.d_data(), d_W.rows(), d_W.cols(), d_lastOutput.cols());
		d_Activate(d_A.d_data(), d_A.size(), network->GetParams().layerActivations[i]);
		d_A.UpdateHostData();
		lastOutput = *d_A.h_matrix();
		cache.A[i+1] = lastOutput;
	}
	return lastOutput;
}

float d_NetTrainer::CalcCost(const MatrixXf h, MatrixXf Y) {
	float coeff = 1.f / Y.cols();
	float sumSqrW = 0.f;
	for(int w = 0; w < (int)network->GetParams().W.size() - 1; ++w) {
		sumSqrW += network->GetParams().W[w].array().pow(2).sum();
	}
	float regCost = 0.5f * float((trainParams.regTerm*trainParams.learningMod) * (sumSqrW / (2.f * (float)trainLabels->cols())));
	return ((Y - h).array().pow(2).sum() * coeff) + regCost;
}

void d_NetTrainer::BackwardPropagation() {
	float m = (float)trainLabels->cols();
	float coeff = float(1.f / m);
	MatrixXf dZ = MatrixXf(cache.A.back() - * trainLabels);
	trainParams.dW.back() = coeff * (dZ * cache.A[cache.A.size() - 2].transpose());
	trainParams.db.back() = coeff * dZ.rowwise().sum();
	for(int l = (int)network->GetParams().layerActivations.size() - 2; l >= 0; --l) {
		MatrixXf lowerA =  cache.A[l];
		switch(network->GetParams().layerActivations[l]) {
		case Sigmoid:
			dZ = BackSigmoid(dZ, l + 1);
			break;
		case Tanh:
			dZ = BackTanh(dZ, l + 1);
			break;
		case ReLU:
			dZ = BackReLU(dZ, l + 1);
			break;
		case LReLU:
			dZ = BackLReLU(dZ, l + 1);
			break;
		default:
			break;
		}
		trainParams.dW[l] = coeff * MatrixXf((dZ * lowerA.transpose()).array() + (0.5f * (trainParams.regTerm*trainParams.learningMod) * network->GetParams().W[l]).array());
		trainParams.db[l] = coeff * dZ.rowwise().sum();
	}

}

void d_NetTrainer::UpdateParameters() {
	for(int i = 0; i < (int)trainParams.dW.size(); ++i) {
		network->GetParams().W[i] -= ((trainParams.learningRate*trainParams.learningMod) * trainParams.dW[i]);
		network->GetParams().b[i] -= ((trainParams.learningRate*trainParams.learningMod) * trainParams.db[i]);
	}
}

void d_NetTrainer::CleanUpAll() {
}


#define BETA1 0.9
#define BETA2 (1.f - FLT_EPSILON)
void d_NetTrainer::UpdateParametersADAM() {
	for(int i = 0; i < (int)trainParams.dW.size(); ++i) {
		d_NetTrainParameters vCorrected = momentum;
		d_NetTrainParameters sCorrected = momentumSqr;
		momentum.dW[i] = BETA1 * momentum.dW[i] + (1 - BETA1) * trainParams.dW[i];
		momentum.db[i] = BETA1 * momentum.db[i] + (1 - BETA1) * trainParams.db[i];
		vCorrected.dW[i] = momentum.dW[i] / (1 - pow(BETA1, 2));
		vCorrected.db[i] = momentum.db[i] / (1 - pow(BETA1, 2));
		momentumSqr.dW[i] = (BETA2 * momentumSqr.dW[i]) + ((1 - BETA2) * MatrixXf(trainParams.dW[i].array().pow(2)));
		momentumSqr.db[i] = (BETA2 * momentumSqr.db[i]) + ((1 - BETA2) * MatrixXf(trainParams.db[i].array().pow(2)));
		sCorrected.dW[i] = momentumSqr.dW[i] / (1 - pow(BETA2, 2));
		sCorrected.db[i] = momentumSqr.db[i] / (1 - pow(BETA2, 2));
		network->GetParams().W[i] -= (trainParams.learningRate*trainParams.learningMod) * MatrixXf(vCorrected.dW[i].array() / (sCorrected.dW[i].array().sqrt() + FLT_EPSILON));
		network->GetParams().b[i] -= (trainParams.learningRate*trainParams.learningMod) * MatrixXf(vCorrected.db[i].array() / (sCorrected.db[i].array().sqrt() + FLT_EPSILON));
	}
}

void d_NetTrainer::UpdateSingleStep() {
	
	cache.cost = CalcCost(ForwardTrain(), *trainLabels);
	BackwardPropagation();
	UpdateParametersADAM();
	

}


