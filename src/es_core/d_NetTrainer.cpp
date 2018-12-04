#include "d_NetTrainer.h"

d_NetTrainer::d_NetTrainer() {
}


d_NetTrainer::d_NetTrainer(Net *net, MatrixXf *data, MatrixXf *labels, float weightScale, float learnRate, float regTerm) {
	network = net;
	trainData = data;
	trainLabels = labels;
	d_trainLabels = d_MatrixXf(*labels);
	trainExamplesCount = (unsigned int)data->cols();
	int nodeCount = 0;
	for(int i = 0; i < (int)network->GetParams().layerSizes.size() - 1; ++i) {
		nodeCount += network->GetParams().layerSizes[i];
		network->GetParams().W[i] *= weightScale;
	}
	trainParams.learningMod = 1.f / nodeCount;
	trainParams.learningRate = learnRate;
	trainParams.regTerm = regTerm;
	cache.d_A.push_back(d_MatrixXf(*trainData));
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
	cache.d_A.push_back(d_MatrixXf( MatrixXf::Zero(A, trainExamplesCount) ));
	cache.d_dZ.push_back(d_MatrixXf(MatrixXf::Zero(A, trainExamplesCount)));
	trainParams.dW.push_back(MatrixXf::Zero(A, B));
	trainParams.db.push_back(MatrixXf::Zero(A, 1));
	momentum.dW.push_back(MatrixXf::Zero(A, B));
	momentum.db.push_back(MatrixXf::Zero(A, 1));
	momentumSqr.dW.push_back(MatrixXf::Zero(A, B));
	momentumSqr.db.push_back(MatrixXf::Zero(A, 1));
}

MatrixXf d_NetTrainer::Visualization(MatrixXf screen) {
	d_MatrixXf d_last = d_MatrixXf(screen);
	for(int i = 0; i < (int)network->GetParams().layerSizes.size() - 1; ++i) {
		d_MatrixXf d_W = d_MatrixXf(network->GetParams().W[i]); //TODO: use device memory
		d_MatrixXf d_b = d_MatrixXf(network->GetParams().b[i]); //TODO: use device memory
		d_MatrixXf d_out = d_MatrixXf(MatrixXf(d_W.rows(), d_last.cols()));
		d_forwardLayer(d_out.d_data(), d_W.d_data(), d_last.d_data(), d_b.d_data(), d_W.rows(), d_W.cols(), d_last.cols());
		d_Activate(d_out.d_data(), d_out.size(), network->GetParams().layerActivations[i]);
		d_out.UpdateHostData();
		d_last = d_out;
	}
	return d_last.h_matrix();
}

d_MatrixXf d_NetTrainer::ForwardTrain() {
	for(int i = 0; i < (int)network->GetParams().layerSizes.size() - 1; ++i) {
		d_MatrixXf d_W = d_MatrixXf(network->GetParams().W[i]); //TODO: use device memory
		d_MatrixXf d_b = d_MatrixXf(network->GetParams().b[i]); //TODO: use device memory
		d_forwardLayer(cache.d_A[i + 1].d_data(), d_W.d_data(), cache.d_A[i].d_data(), d_b.d_data(), d_W.rows(), d_W.cols(), cache.d_A[i].cols());
		d_Activate(cache.d_A[i + 1].d_data(), cache.d_A[i + 1].size(), network->GetParams().layerActivations[i]);		
		cache.d_A[i + 1].UpdateHostData();
	}
	return cache.d_A.back();
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
	vector<MatrixXf> diffs;
	float m = (float)trainLabels->cols();
	float coeff = float(1.f / m);
	MatrixXf dZ = MatrixXf(cache.d_A.back().h_matrix() - *trainLabels);
	//d_subtract(cache.d_dZ.back().d_data(), cache.d_A.back().d_data(), d_trainLabels.d_data(), cache.d_dZ.back().size());
	//cache.d_dZ.back().UpdateHostData();
	//diffs.push_back(cache.d_dZ.back().h_matrix().array() - dZ.array());

	trainParams.dW.back() = coeff * (dZ * cache.d_A[cache.d_A.size() - 2].h_matrix().transpose());
	trainParams.db.back() = coeff * dZ.rowwise().sum();
	for(int l = (int)network->GetParams().layerActivations.size() - 2; l >= 0; --l) {
		MatrixXf lowerA = cache.d_A[l].h_matrix();
		//d_MatrixXf d_W = d_MatrixXf(network->GetParams().W[l].transpose());
		//int m = (int)d_W.rows();
		//int n = (int)d_W.cols();
		//int k = (int)dZ.cols();
		switch(network->GetParams().layerActivations[l]) {
		case Sigmoid:
			dZ = BackSigmoid(dZ, l + 1);
			break;
		case Tanh:
			dZ = BackTanh(dZ, l + 1);
			//d_BackTanh(cache.d_dZ[l].d_data(), d_W.d_data(), cache.d_dZ[l + 1].d_data(), cache.d_A[l].d_data(), m, n, k );
			//cache.d_dZ[l].UpdateHostData();
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

		//MatrixXf expected = network->GetParams().W[l].transpose() * dZ;
		//MatrixXf diff = expected.array() - cache.d_dZ[l].h_matrix().array();

		trainParams.dW[l] = coeff * MatrixXf((dZ * lowerA.transpose()).array() +(0.5f * (trainParams.regTerm*trainParams.learningMod) * network->GetParams().W[l]).array());
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
	cache.cost = CalcCost(ForwardTrain().h_matrix(), *trainLabels);
	BackwardPropagation();
	UpdateParametersADAM();
	

}


