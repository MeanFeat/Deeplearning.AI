#include "es_core_pch.h"
#include "d_NetTrainer.h"
using namespace Eigen;
using namespace std;
static cudaStream_t cuda_stream;
cudaEvent_t start, stop;
d_Matrix d_NetTrainer::to_device(MatrixXf matrix) {
	d_mathInit();
	return d_Matrix(matrix.data(), int(matrix.rows()), int(matrix.cols()));
}
MatrixXf d_NetTrainer::to_host(d_Matrix d_matrix) {
	MatrixXf out = MatrixXf(d_matrix.rows(), d_matrix.cols());
	d_check(cudaMemcpyAsync(out.data(), d_matrix.d_data(), d_matrix.memSize(), cudaMemcpyDeviceToHost));
	return out;
}
d_NetTrainer::d_NetTrainer(): network(nullptr), cache(), trainParams(), d_Buffer(nullptr), profiler() {}
void d_NetTrainer::free()
{
	trainParams.clear();
	cache.clear();
	derivative.clear();
	momentum.clear();
	momentumSqr.clear();
	d_check(cudaFree(cache.d_cost));
}
d_NetTrainer::~d_NetTrainer()
{
	cudaStreamDestroy(cuda_stream);
	free();
}
d_NetTrainParameters &d_NetTrainer::GetTrainParams(){
	return trainParams;
}
d_NetCache &d_NetTrainer::GetCache(){
	return cache;
}
void d_NetTrainer::RefreshHostNetwork() const {
	for (int i = 0; i < trainParams.d_W.size(); ++i) {
		network->GetParams().W[i] = to_host(trainParams.d_W[i]);
		network->GetParams().b[i] = to_host(trainParams.d_b[i]);
	}
}
const d_NetProfiler *d_NetTrainer::GetProfiler() const {
	return &profiler;
}
d_NetTrainer::d_NetTrainer(Net *net, const MatrixXf &data, const MatrixXf &labels, float weightScale, float learnRate, float regTerm) {
	assert(net->GetNodeCount());
	assert(data.size());
	assert(labels.size());
#if _PROFILE
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
#endif
	cudaStreamCreate(&cuda_stream);
	network = net;
	cache.d_A.emplace_back(to_device(data));
	cache.d_AT.emplace_back(to_device(data.transpose()));
	d_trainLabels = to_device(labels);
	trainParams.trainExamplesCount =  uint(data.cols());
	trainParams.coefficient = 1.f / float(trainParams.trainExamplesCount);
	if (network->GetSumOfWeights() == 0.f) {
		network->RandomInit(weightScale);
	}
	for (int i = 0; i < network->GetDepth(); ++i) {
		MatrixXf& W = network->GetParams().W[i];
		MatrixXf& b = network->GetParams().b[i];
		trainParams.d_W.emplace_back(W.data(), int(W.rows()), int(W.cols()));
		trainParams.d_b.emplace_back(b.data(), int(b.rows()), int(b.cols()));
	}
	trainParams.learnRate = learnRate;
	trainParams.learnCoeff = 1.f / float(network->GetNodeCount());
	trainParams.learnMult = trainParams.learnRate*trainParams.learnCoeff;
	trainParams.regTerm = regTerm;
	trainParams.regMod = trainParams.regTerm / float(network->GetNodeCount());
	trainParams.regMult = float(trainParams.regTerm * trainParams.learnCoeff);
	for (int h = 1; h < (int)network->GetDepth() + 1; ++h) {
		AddLayer((int)network->GetParams().layerSizes[h], (int)network->GetParams().layerSizes[h - 1]);
	}
	d_check(cudaMalloc(&cache.d_cost, sizeof(float)));
	d_check(cudaMallocHost(VOID_PTR(&cache.cost), sizeof(float)));
}
void d_NetTrainer::AddLayer(int A, int B) {
	cache.d_A.emplace_back(A, GetTrainExamplesCount());
	cache.d_AT.emplace_back(GetTrainExamplesCount(), A);
	cache.d_dZ.emplace_back(A, GetTrainExamplesCount());
	derivative.d_dW.emplace_back(A, B);
	derivative.d_db.emplace_back(A, 1);
	momentum.d_dW.emplace_back(A, B);
	momentum.d_db.emplace_back(A, 1);
	momentumSqr.d_dW.emplace_back(A, B);
	momentumSqr.d_db.emplace_back(A, 1);
}
void d_NetTrainer::BuildVisualization(const MatrixXf &screen, int * buffer, const int m, const int k) {
	const int size = m*k;
	d_check(cudaMalloc(&d_Buffer, size * sizeof(int)));
	d_VisualA.push_back(to_device(screen));
	for (int i = 0; i < network->GetDepth(); ++i) {
		d_VisualA.emplace_back(trainParams.d_W[i].rows(), d_VisualA[i].cols());
	}
}
void d_NetTrainer::Visualization(int *buffer, const int m, const int k, const bool discrete) {
	d_profile(start, stop, &profiler.visualizationTime,
		for (int i = 0; i < network->GetDepth(); ++i) {
			d_forwardLayer(&d_VisualA[i + 1], &trainParams.d_W[i], &d_VisualA[i], &trainParams.d_b[i]);
			d_activate(&d_VisualA[i + 1], network->GetParams().layerActivations[i]);
		}
	d_drawPixels(d_Buffer, m, k, d_VisualA.back().d_data(), discrete);
	d_check(cudaMemcpyAsync(buffer, d_Buffer, m*k * sizeof(int), cudaMemcpyDeviceToHost, cuda_stream));
	); //d_profile
}
d_Matrix d_NetTrainer::Forward(const d_Matrix &Input) const {
	d_Matrix PreviousLayer = Input;
	for (int i = 0; i < network->GetDepth(); ++i) {
		d_Matrix Layer(trainParams.d_W[i].rows(), PreviousLayer.cols() );
		d_forwardLayer(&Layer, &trainParams.d_W[i], &PreviousLayer, &trainParams.d_b[i]);
		d_activate(&Layer, network->GetParams().layerActivations[i]);
		PreviousLayer = Layer;
	}
	return PreviousLayer;
}
void d_NetTrainer::ForwardTrain() {
	for (int i = 0; i < network->GetDepth(); ++i) {
		d_forwardLayer(&cache.d_A[i + 1], &trainParams.d_W[i], &cache.d_A[i], &trainParams.d_b[i]);
		d_activate(&cache.d_A[i + 1], network->GetParams().layerActivations[i]);
		d_transpose(&cache.d_AT[i + 1], &cache.d_A[i + 1]);
	}
}
float d_NetTrainer::CalcCost(const d_Matrix& Test, const d_Matrix& Labels) const {
	float *d_cost;
	float cost;
	d_check(cudaMalloc(&d_cost, sizeof(float)));
	d_Matrix Error = Test;
	d_subtract_elem(&Error, Test, Labels);
	d_calcCost(d_cost, &Error, &trainParams.d_W, GetRegMultiplier(), 1.f / float(Labels.cols()), float(trainParams.trainExamplesCount)); d_catchErr();
	d_check(cudaMemcpyAsync(&cost, d_cost, sizeof(float), cudaMemcpyDeviceToHost));
	d_check(cudaFree(d_cost));
	Error.free();
	return cost;
}
void d_NetTrainer::CalcCost() {
	d_calcCost(cache.d_cost, &cache.d_dZ.back(), &trainParams.d_W, GetRegMultiplier(), GetCoeff(), float(trainParams.trainExamplesCount)); d_catchErr();
	// TODO: Set this to copy in batches
	d_check(cudaMemcpyAsync(&cache.cost, cache.d_cost, sizeof(float), cudaMemcpyDeviceToHost)); 
}
void d_NetTrainer::BackwardPropagation() {
	d_subtract_elem(&cache.d_dZ.back(), cache.d_A.back(), d_trainLabels);
	d_set_dW(&derivative.d_dW.back(), &cache.d_dZ.back(), &cache.d_AT[cache.d_A.size() - 2], GetCoeff());
	d_set_db(&derivative.d_db.back(), &cache.d_dZ.back(), GetCoeff());
	for (int l = int(network->GetParams().layerActivations.size() - 2); l >= 0; --l) {
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
	for (int i = 0; i < int(derivative.d_dW.size()); ++i) {
		d_updateParameter(&trainParams.d_W[i], &derivative.d_dW[i], trainParams.learnMult);
		d_updateParameter(&trainParams.d_b[i], &derivative.d_db[i], trainParams.learnMult);
	}
}
void d_NetTrainer::UpdateParametersADAM() {
	for (int i = 0; i < int(derivative.d_dW.size()); ++i) {
		d_updateParameterADAM(&trainParams.d_W[i], &derivative.d_dW[i], &momentum.d_dW[i], &momentumSqr.d_dW[i], trainParams.learnMult);
		d_updateParameterADAM(&trainParams.d_b[i], &derivative.d_db[i], &momentum.d_db[i], &momentumSqr.d_db[i], trainParams.learnMult);
	}
}
void d_NetTrainer::TrainSingleEpoch() {
	d_profile(start, stop, &profiler.forwardTime,	ForwardTrain());			d_catchErr();
	d_profile(start, stop, &profiler.backpropTime,	BackwardPropagation());		d_catchErr();
	d_profile(start, stop, &profiler.updateTime,	UpdateParametersADAM());	d_catchErr();
	d_profile(start, stop, &profiler.calcCostTime,	CalcCost());				d_catchErr();
}