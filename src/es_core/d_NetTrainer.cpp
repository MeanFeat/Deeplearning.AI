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
}

d_NetTrainer::~d_NetTrainer() {
}

MatrixXf d_NetTrainer::BackSigmoid(const MatrixXf dZ, int index) {
	MatrixXf A = to_host(cache.d_A[index]);
	return (network->GetParams().W[index].transpose() * dZ).cwiseProduct(MatrixXf::Ones(A.rows(), A.cols()) - A);
} //TODO: remove

MatrixXf d_NetTrainer::BackTanh(const MatrixXf dZ, int index) {
	MatrixXf A = to_host(cache.d_A[index]);
	MatrixXf A1Squared = A.array().pow(2);
	return (network->GetParams().W[index].transpose() * dZ).cwiseProduct(MatrixXf::Ones(A.rows(), A.cols()) - (A1Squared));
}//TODO: remove

MatrixXf d_NetTrainer::BackReLU(const MatrixXf dZ, int index) {
	return (network->GetParams().W[index].transpose() * dZ).cwiseProduct(to_host(cache.d_A[index]).unaryExpr([](float elem) { return elem > 0.f ? 1.f : 0.f; }));
}//TODO: remove

MatrixXf d_NetTrainer::BackLReLU(const MatrixXf dZ, int index) {
	return (network->GetParams().W[index].transpose() * dZ).cwiseProduct(to_host(cache.d_A[index]).unaryExpr([](float elem) { return elem > 0.f ? 1.f : 0.01f; }));
} //TODO: remove

d_NetTrainParameters d_NetTrainer::GetTrainParams() {
	return trainParams;
}

d_NetCache d_NetTrainer::GetCache() {
	return cache;
}

void d_NetTrainer::AddLayer(int A, int B, float weightScale) {
	cache.d_A.push_back(to_device(MatrixXf::Zero(A, trainExamplesCount) ));
	trainParams.dW.push_back(MatrixXf::Zero(A, B));
	trainParams.db.push_back(MatrixXf::Zero(A, 1));
	momentum.dW.push_back(MatrixXf::Zero(A, B));
	momentum.db.push_back(MatrixXf::Zero(A, 1));
	momentumSqr.dW.push_back(MatrixXf::Zero(A, B));
	momentumSqr.db.push_back(MatrixXf::Zero(A, 1));
}

MatrixXf d_NetTrainer::Visualization(MatrixXf screen) {
	d_MatrixXf d_last = to_device(screen);
	d_MatrixXf d_out;
	for(int i = 0; i < (int)network->GetParams().layerSizes.size() - 1; ++i) {
		d_MatrixXf d_W = to_device(network->GetParams().W[i]); //TODO: use device memory
		d_MatrixXf d_b = to_device(network->GetParams().b[i]); //TODO: use device memory
		d_out = to_device(MatrixXf(d_W.rows(), d_last.cols()));
		d_forwardLayer(d_out.d_data(), d_W.d_data(), d_last.d_data(), d_b.d_data(), d_W.rows(), d_W.cols(), d_last.cols());
		d_Activate(d_out.d_data(), d_out.size(), network->GetParams().layerActivations[i]);
		d_last = d_out;
		d_W.free();
		d_b.free();
	}
	MatrixXf out = to_host(d_last);
	d_last.free();
	d_out.free();
	return out;
}

d_MatrixXf d_NetTrainer::ForwardTrain() {
	for(int i = 0; i < (int)network->GetParams().layerSizes.size() - 1; ++i) {
		d_MatrixXf d_W = to_device(network->GetParams().W[i]); //TODO: use device memory
		d_MatrixXf d_b = to_device(network->GetParams().b[i]); //TODO: use device memory
		d_forwardLayer(cache.d_A[i + 1].d_data(), d_W.d_data(), cache.d_A[i].d_data(), d_b.d_data(), d_W.rows(), d_W.cols(), cache.d_A[i].cols());
		d_Activate(cache.d_A[i + 1].d_data(), cache.d_A[i + 1].size(), network->GetParams().layerActivations[i]);
		d_W.free();
		d_b.free();
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

	MatrixXf dZ = MatrixXf(to_host(cache.d_A.back()).array() - trainLabels->array());
	d_subtract(cache.d_dZ.back().d_data(), cache.d_A.back().d_data(), d_trainLabels.d_data(), cache.d_dZ.back().size());
	trainParams.dW.back() = coeff * (dZ * to_host(cache.d_A[cache.d_A.size() - 2]).transpose());
	trainParams.db.back() = coeff * dZ.rowwise().sum();
	for(int l = (int)network->GetParams().layerActivations.size() - 2; l >= 0; --l) {
		d_MatrixXf d_W = to_device(network->GetParams().W[l + 1]); //TODO: replace with device memory
		switch(network->GetParams().layerActivations[l]) {
		case Sigmoid:
			dZ = BackSigmoid(dZ, l + 1);
			break;
		case Tanh:
			d_BackTanh(cache.d_dZ[l].d_data(), d_W.d_data(), cache.d_dZ[l + 1].d_data(), cache.d_A[l+1].d_data(), (int)d_W.cols(), (int)d_W.rows(), (int)dZ.cols());		
			dZ = BackTanh(dZ, l + 1);
			diffs.push_back(to_host(cache.d_dZ[l]).array() - (dZ).array());
			break;
		case ReLU:
			//d_BackReLU(cache.d_dZ[l].d_data(), d_W.d_data(), cache.d_dZ[l + 1].d_data(), cache.d_A[l + 1].d_data(), (int)d_W.cols(), (int)d_W.rows(), (int)dZ.cols());
			//cache.d_dZ[l].UpdateHostData();
			dZ = BackReLU(dZ, l + 1);
			break;
		case LReLU:
			dZ = BackLReLU(dZ, l + 1);
			break;
		default:
			break;

		}

		d_W.free();
		trainParams.dW[l] = coeff * MatrixXf((dZ * to_host(cache.d_A[l]).transpose()).array()
											 + (0.5f * (trainParams.regTerm*trainParams.learningMod) * network->GetParams().W[l]).array());
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
	cache.cost = CalcCost(to_host(ForwardTrain()), *trainLabels);
	BackwardPropagation();
	UpdateParametersADAM();

	/*d_MatrixXf A = d_MatrixXf(MatrixXf::Random(8, 5));
	d_MatrixXf B = d_MatrixXf(MatrixXf::Random(5, 777));
	d_MatrixXf C = d_MatrixXf(MatrixXf::Zero(8, 777));
	MatrixXf expected = A.h_matrix()*B.h_matrix();
	MatrixXf diff = C.h_matrix().array() - expected.array();
	return;*/
}


