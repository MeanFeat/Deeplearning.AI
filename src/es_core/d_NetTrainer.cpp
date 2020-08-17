#include "es_core_pch.h"
#include "d_NetTrainer.h"
using namespace Eigen;
using namespace std;
static cudaStream_t cuda_stream;
cudaEvent_t start, stop;
d_Matrix to_device(MatrixXf matrix) {
	d_mathInit();
	//transpose data only to Column Major
	MatrixXf temp = matrix.transpose();
	return d_Matrix(temp.data(), (int)matrix.rows(), (int)matrix.cols());
}
MatrixXf to_host(d_Matrix d_matrix) {
	// return to Row Major order
	MatrixXf out = MatrixXf(d_matrix.cols(), d_matrix.rows());
	d_check(cudaMemcpy(out.data(), d_matrix.d_data(), d_matrix.memSize(), cudaMemcpyDeviceToHost));
	return out.transpose();
}
d_NetTrainer::d_NetTrainer() {}
d_NetTrainer::~d_NetTrainer() {}

d_NetTrainParameters d_NetTrainer::GetTrainParams() {
	return trainParams;
}
d_NetCache d_NetTrainer::GetCache() {
	return cache;
}

void d_NetTrainer::RefreshHostNetwork() {
	for (int i = 0; i < trainParams.d_W.size(); ++i) {
		network->GetParams().W[i] = to_host(trainParams.d_W[i]);
		network->GetParams().b[i] = to_host(trainParams.d_b[i]);
	}
}
d_NetProfiler d_NetTrainer::GetProfiler() {
	return profiler;
}
d_NetTrainer::d_NetTrainer(Net *net, const MatrixXf &data, const MatrixXf &labels, float weightScale, float learnRate, float regTerm) {
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaStreamCreate(&cuda_stream);
	network = net;
	cache.d_A.push_back(to_device(data));
	cache.d_AT.push_back(to_device(data.transpose()));
	d_trainLabels = to_device(labels);
	trainParams.trainExamplesCount = (unsigned int)data.cols();
	trainParams.coefficiant = 1.f / (float)trainParams.trainExamplesCount;
	int nodeCount = 0;
	for (int i = 0; i < network->GetDepth(); ++i) {
		nodeCount += network->GetParams().layerSizes[i];
		trainParams.d_W.push_back(to_device(network->GetParams().W[i] * weightScale));
		trainParams.d_b.push_back(to_device(network->GetParams().b[i]));
	}
	trainParams.learnRate = learnRate;
	trainParams.learnCoeff = 1.f / (float)nodeCount;
	trainParams.learnMult = trainParams.learnRate*trainParams.learnCoeff;
	trainParams.regTerm = regTerm;
	trainParams.regMod = trainParams.regTerm / (float)nodeCount;
	trainParams.regMult = float((trainParams.regTerm * trainParams.learnCoeff));
	for (int h = 1; h < (int)network->GetDepth() + 1; ++h) {
		AddLayer((int)network->GetParams().layerSizes[h], (int)network->GetParams().layerSizes[h - 1]);
	}
}
void d_NetTrainer::AddLayer(int A, int B) {
	cache.d_A.push_back(d_Matrix(A, GetTrainExamplesCount()));
	cache.d_AT.push_back(d_Matrix(GetTrainExamplesCount(), A));
	cache.d_dZ.push_back(d_Matrix(A, GetTrainExamplesCount()));
	derivative.d_dW.push_back(d_Matrix(A, B));
	derivative.d_db.push_back(d_Matrix(A, 1));
	momentum.d_dW.push_back(d_Matrix(A, B));
	momentum.d_db.push_back(d_Matrix(A, 1));
	momentumSqr.d_dW.push_back(d_Matrix(A, B));
	momentumSqr.d_db.push_back(d_Matrix(A, 1));
}
void d_NetTrainer::BuildVisualization(const MatrixXf &screen, int * buffer, int m, int k) {
	d_check(cudaMalloc((void **)&d_Buffer, m*k * sizeof(int)));
	d_VisualA.push_back(to_device(screen));
	for (int i = 0; i < network->GetDepth(); ++i) {
		d_VisualA.push_back(d_Matrix(trainParams.d_W[i].rows(), d_VisualA[i].cols()));
	}
}
void d_NetTrainer::Visualization(int *buffer, int m, int k, bool discrete) {
	d_profile(start, stop, &profiler.visualizationTime,
		for (int i = 0; i < network->GetDepth(); ++i) {
			d_forwardLayer(&d_VisualA[i + 1], &trainParams.d_W[i], &d_VisualA[i], &trainParams.d_b[i]);
			d_activate(&d_VisualA[i + 1], network->GetParams().layerActivations[i]);
		}
	d_drawPixels(d_Buffer, m, k, d_VisualA.back().d_data(), discrete);
	cudaMemcpyAsync(buffer, d_Buffer, m*k * sizeof(int), cudaMemcpyDeviceToHost, cuda_stream);
	);
}
void d_NetTrainer::ForwardTrain() {
	for (int i = 0; i < network->GetDepth(); ++i) {
		d_forwardLayer(&cache.d_A[i + 1], &trainParams.d_W[i], &cache.d_A[i], &trainParams.d_b[i]);
		d_activate(&cache.d_A[i + 1], network->GetParams().layerActivations[i]);
		d_transpose(&cache.d_AT[i + 1], &cache.d_A[i + 1]);
	}
}
float d_NetTrainer::CalcCost() {
	float *d_cost;
	cudaMalloc((void**)&d_cost, sizeof(float));
	d_calcCost(d_cost, &cache.d_dZ.back(),
		&trainParams.d_W, GetRegMultipier(), GetCoeff(), (float)trainParams.trainExamplesCount); d_catchErr();
	cudaMemcpyAsync(&cache.cost, d_cost, sizeof(float), cudaMemcpyDeviceToHost, cuda_stream); d_catchErr();
	cudaFree(d_cost); d_catchErr();
	return cache.cost;
}
void d_NetTrainer::BackwardPropagation() {
	d_subtract_elem(&cache.d_dZ.back(), cache.d_A.back(), d_trainLabels);
	d_set_dW(&derivative.d_dW.back(), &cache.d_dZ.back(), &cache.d_A[cache.d_A.size() - 2], GetCoeff());
	d_set_db(&derivative.d_db.back(), &cache.d_dZ.back(), GetCoeff());
	for (int l = (int)network->GetParams().layerActivations.size() - 2; l >= 0; --l) {
		switch (network->GetParams().layerActivations[l]) {
		case Sigmoid:
			d_backSigmoid(&cache.d_dZ[l], &trainParams.d_W[l + 1], &cache.d_dZ[l + 1], &cache.d_A[l + 1]);
			break;
		case Tanh:
			d_backTanh(&cache.d_dZ[l], &trainParams.d_W[l + 1], &cache.d_dZ[l + 1], &cache.d_A[l + 1]);
			break;
		case ReLU:
			d_backReLU(&cache.d_dZ[l], &trainParams.d_W[l + 1], &cache.d_dZ[l + 1], &cache.d_A[l + 1]);
			break;
		case LReLU:
			d_backLReLU(&cache.d_dZ[l], &trainParams.d_W[l + 1], &cache.d_dZ[l + 1], &cache.d_A[l + 1]);
			break;
		case Sine:
			d_backSine(&cache.d_dZ[l], &trainParams.d_W[l + 1], &cache.d_dZ[l + 1], &cache.d_A[l + 1]);
			break;
		case Linear:
		default:
			break;
		}
		d_set_dW_Reg(&derivative.d_dW[l], &cache.d_dZ[l], &cache.d_AT[l], &trainParams.d_W[l], GetCoeff(), 0.5f * trainParams.regMod);
		d_set_db(&derivative.d_db[l], &cache.d_dZ[l], GetCoeff());
	}
}
void d_NetTrainer::UpdateParameters() {
	for (int i = 0; i < (int)derivative.d_dW.size(); ++i) {
		d_updateParameter(&trainParams.d_W[i], &derivative.d_dW[i], trainParams.learnMult);
		d_updateParameter(&trainParams.d_b[i], &derivative.d_db[i], trainParams.learnMult);
	}
}
void d_NetTrainer::UpdateParametersADAM() {
	for (int i = 0; i < (int)derivative.d_dW.size(); ++i) {
		d_updateParameterADAM(&trainParams.d_W[i], &derivative.d_dW[i], &momentum.d_dW[i], &momentumSqr.d_dW[i], trainParams.learnMult);
		d_updateParameterADAM(&trainParams.d_b[i], &derivative.d_db[i], &momentum.d_db[i], &momentumSqr.d_db[i], trainParams.learnMult);
	}
}
void d_NetTrainer::TrainSingleEpoch() {
	d_profile(start, stop, &profiler.forwardTime, ForwardTrain()); d_catchErr();
	d_profile(start, stop, &profiler.backpropTime, BackwardPropagation()); d_catchErr();
	d_profile(start, stop, &profiler.updateTime, UpdateParametersADAM()); d_catchErr();
	d_profile(start, stop, &profiler.calcCostTime, CalcCost()); d_catchErr();
}