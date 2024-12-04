#include "es_core_pch.h"
#include "d_NetTrainer.h"
#include <numeric>
#include <random>
using namespace Eigen;
using namespace std;
static cudaStream_t cuda_stream_default;
static cudaStream_t cuda_stream_load;
static cudaStream_t cuda_stream_cublas;
cudaEvent_t start, stop;
d_Matrix d_NetTrainer::to_device(MatrixXf matrix) {
	d_mathInit();
	d_Matrix d_matrix = d_Matrix(int(matrix.rows()), int(matrix.cols()));
	d_check(cublasSetMatrixAsync(matrix.rows(), matrix.cols(), sizeof(float), matrix.data(), matrix.rows(), d_matrix.d_data(), matrix.rows(), cuda_stream_load));
	return d_matrix;
}
MatrixXf d_NetTrainer::to_host(d_Matrix d_matrix) {
	MatrixXf out = MatrixXf(d_matrix.rows(), d_matrix.cols());
	d_check(cublasGetMatrixAsync(d_matrix.rows(), d_matrix.cols(), sizeof(float), d_matrix.d_data(), d_matrix.rows(), out.data(), d_matrix.rows(), cuda_stream_load));
	return out;
}
d_NetBatchTrainingData::d_NetBatchTrainingData(const MatrixXf& data, const MatrixXf& labels) {
	d_Data.setShape(int(data.rows()), int(data.cols()));
	d_Labels.setShape(int(labels.rows()), int(labels.cols()));
	d_check(cudaMallocHost(reinterpret_cast<void**>(&d_Data.d_data()), d_Data.memSize()));
	d_check(cudaMallocHost(reinterpret_cast<void**>(&d_Labels.d_data()), d_Labels.memSize()));
	d_check(cudaMemcpyAsync(d_Data.d_data(), data.data(), d_Data.memSize(), cudaMemcpyHostToHost, cuda_stream_load));
	d_check(cudaMemcpyAsync(d_Labels.d_data(), labels.data(), d_Labels.memSize(), cudaMemcpyHostToHost, cuda_stream_load));
}
d_NetTrainer::d_NetTrainer(): network(nullptr), cache(), trainParams(), d_Buffer(nullptr), profiler(), batchDataPool()
{
	d_mathInit();
	cudaStreamCreate(&cuda_stream_default);
	cudaStreamCreate(&cuda_stream_load);
	cudaStreamCreate(&cuda_stream_cublas);
}
d_NetTrainer::d_NetTrainer(Net *net, const MatrixXf &data, const MatrixXf &labels, float weightScale, float learnRate, float regTerm, int batchCount) {
	assert(net->GetNodeCount());
	assert(data.size());
	assert(labels.size());
#if _PROFILE
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
#endif
	d_mathInit();
	cudaStreamCreate(&cuda_stream_default);
	cudaStreamCreate(&cuda_stream_load);
	cudaStreamCreate(&cuda_stream_cublas);
	d_check(cublasSetStream(cublasHandle, cuda_stream_cublas));
	d_check(cublasSetMathMode(cublasHandle, CUBLAS_TF32_TENSOR_OP_MATH));
	network = net;
	trainParams.batchCount = batchCount;
	trainParams.trainExamplesCount = uint(data.cols());
	if (batchCount > 1) {
		CreateBatchData(data, labels);
		cache.d_A.emplace_back(int(data.rows()), GetBatchSize());
		d_trainLabels = d_Matrix(int(labels.rows()), GetBatchSize());
		trainParams.regTerm = regTerm / float(trainParams.batchCount);
	}
	else {
		cache.d_A.emplace_back(to_device(data));
		d_trainLabels = to_device(labels);
		trainParams.regTerm = regTerm;
	}
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
	trainParams.regMod = trainParams.regTerm / float(network->GetNodeCount());
	trainParams.regMult = float(trainParams.regTerm * trainParams.learnCoeff);
	for (int h = 1; h < network->GetDepth() + 1; ++h) {
		AddLayer(network->GetParams().layerSizes[h], network->GetParams().layerSizes[h - 1]);
	}
	d_check(cudaMalloc(&cache.d_cost, sizeof(float)));
	d_check(cudaMallocHost(VOID_PTR(&cache.cost), sizeof(float)));
}
d_NetTrainer::~d_NetTrainer()
{
	cudaStreamDestroy(cuda_stream_default);
	cudaStreamDestroy(cuda_stream_load);
	free();
}
void d_NetTrainer::free(){
	trainParams.clear();
	cache.clear();
	derivative.clear();
	momentum.clear();
	momentumSqr.clear();
	batchDataPool.clear();
	d_check(cudaFree(cache.d_cost));
}
void d_NetTrainer::CreateBatchData(const MatrixXf& data, const MatrixXf& labels) {
	shuffledBatchIndices.resize(GetTotalTrainingExamples());
	iota(shuffledBatchIndices.begin(), shuffledBatchIndices.end(), 0);
	batchDataPool = d_NetBatchTrainingData(data, labels);
}
void d_NetTrainer::ShuffleData() {
	random_device rd;
	mt19937 g(rd());
	shuffle(shuffledBatchIndices.begin(), shuffledBatchIndices.end(), g);
}
void d_NetTrainer::LoadBatchData(const int batchIndex) {
	const int start_idx = batchIndex * GetBatchSize();
	const int end_idx =  min(start_idx + GetBatchSize(), GetTotalTrainingExamples());
	if (trainParams.shuffleData) {
		for (int i = start_idx; i < end_idx; ++i) {
			const int idx = shuffledBatchIndices[i];
			const int dataRows = cache.d_A[0].rows();
			const int labelRows = d_trainLabels.rows();
			const int d_idx = (i - start_idx);
			d_check(cudaMemcpyAsync(cache.d_A[0].d_data() + d_idx * dataRows, batchDataPool.d_Data.d_data() + idx * dataRows, dataRows * sizeof(float), cudaMemcpyHostToDevice, cuda_stream_load));
			d_check(cudaMemcpyAsync(d_trainLabels.d_data() + d_idx * labelRows, batchDataPool.d_Labels.d_data() + idx * labelRows, labelRows * sizeof(float), cudaMemcpyHostToDevice, cuda_stream_load));
		}
	}
	else {
		const int dataRows = cache.d_A[0].rows();
		const int labelRows = d_trainLabels.rows();
		d_check(cublasSetMatrixAsync(dataRows, GetBatchSize(), sizeof(float), batchDataPool.d_Data.d_data() + start_idx * dataRows, dataRows, cache.d_A[0].d_data(), dataRows, cuda_stream_load));
		d_check(cublasSetMatrixAsync(labelRows, GetBatchSize(), sizeof(float), batchDataPool.d_Labels.d_data() + start_idx * labelRows, labelRows, d_trainLabels.d_data(), labelRows, cuda_stream_load));
	}
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
void d_NetTrainer::AddLayer(int A, int B) {
	cache.d_A.emplace_back(A, GetBatchSize());
	cache.d_dZ.emplace_back(A, GetBatchSize());
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
	d_check(cudaMemcpyAsync(buffer, d_Buffer, m*k * sizeof(int), cudaMemcpyDeviceToHost, cuda_stream_default));
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
	}
}
float d_NetTrainer::CalcCost(const d_Matrix& Test, const d_Matrix& Labels) const {
	float *d_cost;
	float cost;
	d_check(cudaMalloc(&d_cost, sizeof(float)));
	d_Matrix Error = Test;
	d_subtract_elem(&Error, Test, Labels);
	d_calcCost(d_cost, &Error, &trainParams.d_W, GetRegMultiplier(), 1.f / float(Labels.cols()), float(Test.cols())); d_catchErr();
	d_check(cudaMemcpyAsync(&cost, d_cost, sizeof(float), cudaMemcpyDeviceToHost, cuda_stream_default));
	d_check(cudaFree(d_cost));
	Error.free();
	return cost;
}
void d_NetTrainer::CalcCost() const {
	d_calcCost(cache.d_cost, &cache.d_dZ.back(), &trainParams.d_W, GetRegMultiplier(), GetCoeff(), float(GetTotalTrainingExamples())); d_catchErr();
}
void d_NetTrainer::BackwardPropagation() {
	d_subtract_elem(&cache.d_dZ.back(), cache.d_A.back(), d_trainLabels); d_catchErr();
	d_Matrix d_ATLast(cache.d_A[cache.d_A.size() - 2].cols(), cache.d_A[cache.d_A.size() - 2].rows()); d_catchErr();
	d_transpose(&d_ATLast, &cache.d_A[cache.d_A.size() - 2]); d_catchErr();
	d_set_dW(&derivative.d_dW.back(), &cache.d_dZ.back(), &d_ATLast, GetCoeff()); d_catchErr();
	d_set_db(&derivative.d_db.back(), &cache.d_dZ.back(), GetCoeff()); d_catchErr();
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
		d_Matrix d_AT = d_Matrix(cache.d_A[l].cols(), cache.d_A[l].rows());
		d_transpose(&d_AT, &cache.d_A[l]);
		d_set_dW_Reg(&derivative.d_dW[l], &cache.d_dZ[l], &d_AT, &trainParams.d_W[l], GetCoeff(), 0.5f * trainParams.regMod);
		d_set_db(&derivative.d_db[l], &cache.d_dZ[l], GetCoeff());
		d_AT.free();
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
	if (trainParams.batchCount > 1)	{
		float totalCost = 0.f;
		for (int i = 0; i < trainParams.batchCount; ++i) {
			float batchCost = 0.f;
			d_profile(start, stop, &profiler.loadBatchData, LoadBatchData(i));	d_catchErr();
			d_profile(start, stop, &profiler.forwardTime,	ForwardTrain());			d_catchErr();
			d_profile(start, stop, &profiler.backpropTime,	BackwardPropagation());		d_catchErr();
			d_profile(start, stop, &profiler.updateTime,	UpdateParametersADAM());	d_catchErr();
			d_profile(start, stop, &profiler.calcCostTime,	CalcCost());				d_catchErr();
			d_check(cudaMemcpyAsync(&batchCost, cache.d_cost, sizeof(float), cudaMemcpyDeviceToHost, cuda_stream_default));
			totalCost += batchCost;
		}
		if (trainParams.shuffleData) {
			ShuffleData();
		}
		cudaStreamSynchronize(cuda_stream_load);
		cache.cost = totalCost / float(trainParams.batchCount);
	}
	else
	{
		d_profile(start, stop, &profiler.forwardTime,	ForwardTrain());			d_catchErr();
		d_profile(start, stop, &profiler.backpropTime,	BackwardPropagation());		d_catchErr();
		d_profile(start, stop, &profiler.updateTime,	UpdateParametersADAM());	d_catchErr();
		d_profile(start, stop, &profiler.calcCostTime,	CalcCost());				d_catchErr();
		d_check(cudaMemcpyAsync(&cache.cost, cache.d_cost, sizeof(float), cudaMemcpyDeviceToHost, cuda_stream_default)); 
	}
}