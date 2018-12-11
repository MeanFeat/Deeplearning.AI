#include "d_NetTrainer.h"

using namespace Eigen;

d_Matrix to_device(MatrixXd matrix) {
	return d_Matrix(matrix.data(), (int)matrix.rows(), (int)matrix.cols());
}

MatrixXd to_host(d_Matrix d_matrix) {
	MatrixXd out = MatrixXd(d_matrix.rows(), d_matrix.cols());
	cudaMemcpy(out.data(), d_matrix.d_data(), d_matrix.memSize(), cudaMemcpyDeviceToHost);
	return out;
}

d_NetTrainer::d_NetTrainer() {
}


d_NetTrainer::d_NetTrainer(Net *net, MatrixXd *data, MatrixXd *labels, double weightScale, double learnRate, double regTerm) {
	network = net;
	trainData = data;
	trainLabels = labels;
	d_trainLabels = to_device(*labels);
	trainParams.trainExamplesCount = (unsigned int)data->cols();
	trainParams.coefficiant = 1.0 / (double)trainParams.trainExamplesCount;
	int nodeCount = 0;
	for(int i = 0; i < (int)network->GetParams().layerSizes.size() - 1; ++i) {
		nodeCount += network->GetParams().layerSizes[i];
		trainParams.d_W.push_back(to_device(network->GetParams().W[i] * weightScale));
		trainParams.d_b.push_back(to_device(network->GetParams().b[i]));
	}
	trainParams.learningRate = learnRate;
	trainParams.learningMod = learnRate / (double)nodeCount;
	trainParams.regTerm = regTerm;
	trainParams.regMod = regTerm / (double)nodeCount;
	cache.d_A.push_back(to_device(*trainData));
	for(int h = 1; h < (int)network->GetParams().layerSizes.size(); ++h) {
		AddLayer((int)network->GetParams().layerSizes[h], (int)network->GetParams().layerSizes[h - 1]);
	}
	for(int i = 1; i < cache.d_A.size(); ++i)	{
		cache.d_dZ.push_back(to_device(MatrixXd::Zero(cache.d_A[i].rows(), TrainExamplesCount())));
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

void d_NetTrainer::AddLayer(int A, int B) {
	cache.d_A.push_back(to_device(MatrixXd::Zero(A, TrainExamplesCount())));
	trainParams.d_dW.push_back(to_device(MatrixXd::Zero(A, B)));
	trainParams.d_db.push_back(to_device(MatrixXd::Zero(A, 1)));
	momentum.d_dW.push_back(to_device(MatrixXd::Zero(A, B)));
	momentum.d_db.push_back(to_device(MatrixXd::Zero(A, 1)));
	momentumSqr.d_dW.push_back(to_device(MatrixXd::Zero(A, B)));
	momentumSqr.d_db.push_back(to_device(MatrixXd::Zero(A, 1)));
}

void d_NetTrainer::BuildVisualization(MatrixXd screen, int * buffer, int m, int k) {
	cudaMalloc((void **)&d_Buffer, m*k * sizeof(int));
	d_VisualA.push_back(to_device(screen));
	for(int i = 0; i < (int)network->GetParams().layerSizes.size() - 1; ++i) {
		d_VisualA.push_back(to_device(MatrixXd(trainParams.d_W[i].rows(), d_VisualA[i].cols())));
	}
}

void d_NetTrainer::Visualization(int * buffer, int m, int k, bool discrete, cudaStream_t *stream) {
	for(int i = 0; i < (int)network->GetParams().layerSizes.size() - 1; ++i) {
		d_forwardLayer(&d_VisualA[i+1], &trainParams.d_W[i], &d_VisualA[i], &trainParams.d_b[i]);
		d_Activate(&d_VisualA[i + 1], network->GetParams().layerActivations[i]);
	}
	d_drawPixels(d_Buffer, m,k, d_VisualA.back().d_data(), discrete);
	cudaMemcpyAsync(buffer, d_Buffer, m*k * sizeof(int), cudaMemcpyDeviceToHost, *stream);
}

void d_NetTrainer::UpdateNetwork() {
	for(int i = 0; i < trainParams.d_W.size(); ++i) {
		network->GetParams().W[i] = to_host(trainParams.d_W[i]);
		network->GetParams().b[i] = to_host(trainParams.d_b[i]);
	}
}

void d_NetTrainer::ForwardTrain() {
	for(int i = 0; i < (int)network->GetParams().layerSizes.size() - 1; ++i) {
		d_forwardLayer(&cache.d_A[i + 1], &trainParams.d_W[i], &cache.d_A[i], &trainParams.d_b[i]);
		d_Activate(&cache.d_A[i + 1], network->GetParams().layerActivations[i]);
	}
}

double d_NetTrainer::CalcCost() {
	double sumSqrW = 0.0;
	for(int w = 0; w < (int)trainParams.d_W.size() - 1; ++w) {
		sumSqrW += to_host(trainParams.d_W[w]).array().pow(2).sum();
	}
	double regCost = 0.5 * double((trainParams.regMod) * (sumSqrW / (2.0 * (double)TrainExamplesCount())));
	return ((*trainLabels - to_host(cache.d_A.back())).array().pow(2).sum() * Coeff()) + regCost;
}

void d_NetTrainer::BackwardPropagation() {
	d_subtract(&cache.d_dZ.back(), &cache.d_A.back(), &d_trainLabels);
	d_Set_dW(&trainParams.d_dW.back(), &cache.d_dZ.back(), &cache.d_A[cache.d_A.size() - 2], Coeff());
	d_Set_db(&trainParams.d_db.back(), &cache.d_dZ.back(), Coeff());
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
		d_Set_dW(&trainParams.d_dW[l], &cache.d_dZ[l], &cache.d_A[l], &trainParams.d_W[l], Coeff(), trainParams.regMod);
		d_Set_db(&trainParams.d_db[l], &cache.d_dZ[l], Coeff());
	}
}

void d_NetTrainer::UpdateParameters() {
	for(int i = 0; i < (int)trainParams.d_dW.size(); ++i) {
		d_UpdateParameter(&trainParams.d_W[i], &trainParams.d_dW[i], trainParams.learningMod);
		d_UpdateParameter(&trainParams.d_b[i], &trainParams.d_db[i], trainParams.learningMod);
	}
}

void d_NetTrainer::UpdateParametersADAM() {
	for(int i = 0; i < (int)trainParams.d_dW.size(); ++i) {
		d_UpdateParameterADAM(&trainParams.d_W[i], &trainParams.d_dW[i], &momentum.d_dW[i], &momentumSqr.d_dW[i], trainParams.learningMod);
		d_UpdateParameterADAM(&trainParams.d_b[i], &trainParams.d_db[i], &momentum.d_db[i], &momentumSqr.d_db[i], trainParams.learningMod);
	}
}

void d_NetTrainer::UpdateSingleStep() {
	ForwardTrain();
	BackwardPropagation();
	UpdateParametersADAM();
	cache.cost = CalcCost();
}


