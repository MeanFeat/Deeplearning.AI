#include "d_NetTrainer.h"

using namespace Eigen;

d_MatrixXf to_device(MatrixXf matrix) {
	return d_MatrixXf(matrix.data(), (int)matrix.rows(), (int)matrix.cols());
}

MatrixXf to_host(d_MatrixXf d_matrix) {
	MatrixXf out = MatrixXf(d_matrix.rows(), d_matrix.cols());
	cudaMemcpy(out.data(), d_matrix.d_data(), d_matrix.memSize(), cudaMemcpyDeviceToHost);
	return out;
}

d_NetTrainer::d_NetTrainer() {
}


d_NetTrainer::d_NetTrainer(Net *net, MatrixXf *data, MatrixXf *labels, float weightScale, float learnRate, float regTerm) {
	network = net;
	trainData = data;
	trainLabels = labels;
	d_trainLabels = to_device(*labels);
	trainExamplesCount = (unsigned int)data->cols();
	int nodeCount = 0;
	for(int i = 0; i < (int)network->GetParams().layerSizes.size() - 1; ++i) {
		nodeCount += network->GetParams().layerSizes[i];
		network->GetParams().W[i] *= weightScale;
	}
	trainParams.learningMod = 1.f / nodeCount;
	trainParams.learningRate = learnRate;
	trainParams.regTerm = regTerm;
	cache.d_A.push_back(to_device(*trainData));
	for(int h = 1; h < (int)network->GetParams().layerSizes.size(); ++h) {
		AddLayer((int)network->GetParams().layerSizes[h], (int)network->GetParams().layerSizes[h - 1], weightScale);
	}
	for(int i = 1; i < cache.d_A.size(); ++i)	{
		cache.d_dZ.push_back(to_device(MatrixXf::Zero(cache.d_A[i].rows(), trainExamplesCount)));
	}
	for(int w = 0; w < net->GetParams().W.size(); ++w) {
		trainParams.d_W.push_back(to_device(net->GetParams().W[w]));
		trainParams.d_b.push_back(to_device(net->GetParams().b[w]));
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
	cache.d_A.push_back(to_device(MatrixXf::Zero(A, trainExamplesCount)));
	trainParams.dW.push_back(MatrixXf::Zero(A, B));
	trainParams.d_dW.push_back(to_device(MatrixXf::Zero(A, B)));
	trainParams.db.push_back(MatrixXf::Zero(A, 1));
	trainParams.d_db.push_back(to_device(MatrixXf::Zero(A, 1)));
	momentum.dW.push_back(MatrixXf::Zero(A, B));
	momentum.db.push_back(MatrixXf::Zero(A, 1));
	momentumSqr.dW.push_back(MatrixXf::Zero(A, B));
	momentumSqr.db.push_back(MatrixXf::Zero(A, 1));
}

void d_NetTrainer::Visualization(MatrixXf screen, int * buffer,int m,int k, bool discrete) {
	vector<d_MatrixXf> d_last;
	d_last.push_back(to_device(screen));
	for(int i = 0; i < (int)network->GetParams().layerSizes.size() - 1; ++i) {
		d_last.push_back(to_device(MatrixXf(trainParams.d_W[i].rows(), d_last[i].cols())));
		d_forwardLayer(&d_last[i+1], &trainParams.d_W[i], &d_last[i], &trainParams.d_b[i]);
		d_Activate(&d_last[i + 1], network->GetParams().layerActivations[i]);
	}
	int *d_buffer;
	cudaMalloc((void **)&d_buffer, m*k * sizeof(int));
	d_drawPixels(d_buffer, m,k, d_last.back().d_data(), discrete);
	cudaMemcpy(buffer, d_buffer, m*k * sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(d_buffer);	
	for (int i = 0; i < d_last.size(); ++i){
		d_last[i].free();
	}
}

void d_NetTrainer::ForwardTrain() {
	for(int i = 0; i < (int)network->GetParams().layerSizes.size() - 1; ++i) {
		d_forwardLayer(&cache.d_A[i + 1], &trainParams.d_W[i], &cache.d_A[i], &trainParams.d_b[i]);
		d_Activate(&cache.d_A[i + 1], network->GetParams().layerActivations[i]);
	}
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
	vector<MatrixXf> diffs;//TODO: remove
	float coeff = float(1.f / trainExamplesCount);

	d_subtract(&cache.d_dZ.back(), &cache.d_A.back(), &d_trainLabels);
	trainParams.dW.back() = coeff * (to_host(cache.d_dZ.back()) * to_host(cache.d_A[cache.d_A.size() - 2]).transpose());
	trainParams.db.back() = coeff * to_host(cache.d_dZ.back()).rowwise().sum();

	//d_Set_dW(&trainParams.d_dW.back(), &cache.d_dZ.back(), &cache.d_A[cache.d_A.size() - 2], coeff);
	/*d_matrixMult_rhsT(&trainParams.d_dW.back(), &cache.d_dZ.back(), &cache.d_A[cache.d_A.size() - 2]);

	MatrixXf expected = (to_host(cache.d_dZ.back()) * to_host(cache.d_A[cache.d_A.size() - 2]).transpose());
	MatrixXf test = to_host(trainParams.d_dW.back());
	diffs.push_back(test.array() - expected.array());*/

	for(int l = (int)network->GetParams().layerActivations.size() - 2; l >= 0; --l) {
		switch(network->GetParams().layerActivations[l]) {
		case Sigmoid:
			d_BackSigmoid(&cache.d_dZ[l], &trainParams.d_W[l+1], &cache.d_dZ[l + 1], &cache.d_A[l + 1]);
			break;
		case Tanh:
			d_BackTanh(&cache.d_dZ[l], &trainParams.d_W[l + 1], &cache.d_dZ[l + 1], &cache.d_A[l + 1]);
			break;
		case ReLU:
			d_BackReLU(&cache.d_dZ[l], &trainParams.d_W[l + 1], &cache.d_dZ[l + 1], &cache.d_A[l + 1]);
			break;
		case LReLU:
			d_BackLReLU(&cache.d_dZ[l], &trainParams.d_W[l + 1], &cache.d_dZ[l + 1], &cache.d_A[l + 1]);
			break;
		default:
			break;
		}
		
		trainParams.dW[l] = coeff * MatrixXf((to_host(cache.d_dZ[l]) * to_host(cache.d_A[l]).transpose()).array()
											 + (0.5f * (trainParams.regTerm*trainParams.learningMod) * network->GetParams().W[l]).array());
		trainParams.db[l] = coeff * to_host(cache.d_dZ[l]).rowwise().sum();
	}
}

void d_NetTrainer::UpdateParameters() {
	for(int i = 0; i < (int)trainParams.dW.size(); ++i) {
		network->GetParams().W[i] -= ((trainParams.learningRate*trainParams.learningMod) * trainParams.dW[i]);
		network->GetParams().b[i] -= ((trainParams.learningRate*trainParams.learningMod) * trainParams.db[i]);
		trainParams.d_W[i] = to_device(network->GetParams().W[i]); //TODO: remove
		trainParams.d_b[i] = to_device(network->GetParams().b[i]); //TODO: remove
	}
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
		trainParams.d_W[i] = to_device(network->GetParams().W[i]); //TODO: remove
		trainParams.d_b[i] = to_device(network->GetParams().b[i]); //TODO: remove
	}
}

void d_NetTrainer::UpdateSingleStep() {
	ForwardTrain();
	cache.cost = CalcCost(to_host(cache.d_A.back()), *trainLabels);
	BackwardPropagation();
	UpdateParametersADAM();	
}


