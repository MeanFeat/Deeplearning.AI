#include "stdNetTrainer.h"

//ADAM
static const float b1 = 0.9f;
static const float b2 = 1.f - FLT_EPSILON;
static const float b1Sqr = b1 * b1;
static const float b2Sqr = b2 * b2;
static const float invB1 = 1.f - b1;
static const float invB2 = 1.f - b2;
static const float invBSq1 = 1.f - b1Sqr;
static const float invBSq2 = 1.f - b2Sqr;


NetTrainer::NetTrainer() {
}


NetTrainer::NetTrainer(Net *net, MatrixXf *data, MatrixXf *labels, float weightScale, float learnRate, float regTerm) {
	network = net;
	trainData = data;
	trainLabels = labels;
	coeff = float(1.f/labels->cols());
	int nodeCount = 0;
	for(int i = 0; i < (int)network->GetParams().layerSizes.size() - 1; ++i){
		nodeCount += network->GetParams().layerSizes[i];
		if( network->GetParams().W[i].sum() == 0.f ) {	 //Don't initialize if we already have weights
			MatrixXf *w = &network->GetParams().W[i];
			*w = MatrixXf::Random(w->rows(), w->cols()) * weightScale;
		}
	}
	trainParams.learningMod = 1.f / nodeCount;
	trainParams.learningRate = learnRate;
	trainParams.regTerm = regTerm;
	for(int i = 1; i < (int)network->GetParams().layerSizes.size(); ++i){
		AddLayer((int)network->GetParams().layerSizes[i], (int)network->GetParams().layerSizes[i - 1]);
	}
}

NetTrainer::~NetTrainer() {
}


NetTrainParameters &NetTrainer::GetTrainParams() {
	return trainParams;
}

NetCache &NetTrainer::GetCache() {
	return cache;
}
void NetTrainer::AddLayer(int A, int B){
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
	MatrixXf lastOutput = MatrixXf(trainData->rows(), trainData->cols());
	lastOutput.noalias() = *trainData;
	for(int i = 0; i < (int)network->GetParams().layerSizes.size() - 1; ++i) {
		MatrixXf weighed = network->GetParams().W[i] * lastOutput;
		cache.Z[i].noalias() = ( weighed ).colwise() + (VectorXf)network->GetParams().b[i];
		lastOutput = Net::Activate(network->GetParams().layerActivations[i], cache.Z[i]);
		cache.A[i].noalias() = lastOutput;
	}
	return lastOutput;
}

float NetTrainer::CalcCost(const MatrixXf *h, const MatrixXf *Y) {
	float sumSqrW = 0.f;
	for(int w = 0; w < (int)network->GetParams().W.size() - 1; ++w) {
		sumSqrW += (network->GetParams().W[w].array() * network->GetParams().W[w].array()).sum();
	}
	float regCost = 0.5f * float((trainParams.regTerm*trainParams.learningMod) * (sumSqrW / (2.f * (float)trainLabels->cols())));
	MatrixXf diff = *Y - *h;
	return (( diff.array() * diff.array()).sum() * coeff) + regCost;
}

void NetTrainer::ModifyLearningRate(float m) {
	trainParams.learningRate = max(0.001f, trainParams.learningRate + m);
}

void NetTrainer::ModifyRegTerm(float m) {
	trainParams.regTerm = max(FLT_EPSILON, trainParams.regTerm + m);
}

MatrixXf NetTrainer::BackSigmoid(const MatrixXf &dZ, int index) {
	MatrixXf wT = network->GetParams().W[index + 1].transpose() * dZ;
	return ( wT ).cwiseProduct(MatrixXf::Ones(cache.A[index].rows(), cache.A[index].cols()) - cache.A[index]);
}

MatrixXf NetTrainer::BackTanh(const MatrixXf &dZ, int index) {
	MatrixXf invASqr = 1.f-cache.A[index].array().square();
	MatrixXf product = network->GetParams().W[index + 1].transpose() * dZ;
	return ( product ).cwiseProduct(invASqr);
}

MatrixXf NetTrainer::BackReLU(const MatrixXf &dZ, int index) {
	MatrixXf wT = network->GetParams().W[index + 1].transpose() * dZ;
	return ( wT ).cwiseProduct(cache.A[index].unaryExpr([](float elem) { return elem > 0.f ? 1.f : 0.f; }));
}

MatrixXf NetTrainer::BackLReLU(const MatrixXf &dZ, int index) {
	MatrixXf wT = network->GetParams().W[index + 1].transpose() * dZ;
	return ( wT ).cwiseProduct(cache.A[index].unaryExpr([](float elem) { return elem > 0.f ? 1.f : 0.01f; }));
}

Eigen::MatrixXf NetTrainer::BackSine(const MatrixXf &dZ, int index) {
	MatrixXf wT = network->GetParams().W[index + 1].transpose() * dZ;
	return ( wT ).cwiseProduct(MatrixXf(cache.A[index].array().cos()));
}

MatrixXf NetTrainer::BackActivation(int layerIndex, const MatrixXf &dZ) {
	switch( network->GetParams().layerActivations[layerIndex] ) {
		case Sigmoid:
		return BackSigmoid(dZ, layerIndex);
		break;
		case Tanh:
		return BackTanh(dZ, layerIndex);
		break;
		case ReLU:
		return BackReLU(dZ, layerIndex);
		break;
		case LReLU:
		return BackLReLU(dZ, layerIndex);
		break;
		case Sine:
		return BackSine(dZ, layerIndex);
		break;
		default:
		return dZ;
		break;
	}		
}
void NetTrainer::BackLayer(MatrixXf &dZ, int layerIndex, const MatrixXf *LowerA) {
	dZ = BackActivation(layerIndex, dZ); 
	float lambda = 0.5f * ( trainParams.regTerm * trainParams.learningMod );
	trainParams.dW[layerIndex] = coeff * MatrixXf(( dZ * LowerA->transpose() ).array() + ( lambda * network->GetParams().W[layerIndex] ).array());
	trainParams.db[layerIndex] = dZ.rowwise().sum();
	trainParams.db[layerIndex] *= coeff;
}
void NetTrainer::BackwardPropagation() {
	MatrixXf dZ = MatrixXf(cache.A.back() - *trainLabels);
	trainParams.dW.back() = coeff * (dZ * cache.A[cache.A.size() - 2].transpose());
	trainParams.db.back() = coeff * dZ.rowwise().sum();
	for( int l = (int)network->GetParams().layerActivations.size() - 2; l >= 1; --l ) {
		BackLayer(dZ, l, &cache.A[l - 1]);
	}
	BackLayer(dZ, 0, trainData);
}
void NetTrainer::UpdateParameters() {
	float learnRate = ( trainParams.learningRate*trainParams.learningMod );
	for(int i = 0; i < (int)trainParams.dW.size(); ++i) {
		network->GetParams().W[i] -= learnRate * trainParams.dW[i];
		network->GetParams().b[i] -= learnRate * trainParams.db[i];
	}
}

void NetTrainer::UpdateParametersWithMomentum() {
	float learnRate = ( trainParams.learningRate*trainParams.learningMod );
	for(int i = 0; i < (int)trainParams.dW.size(); ++i) {
		momentum.dW[i] = trainParams.dW[i] + momentum.dW[i].normalized() * cache.cost * 0.025f;
		momentum.db[i] = trainParams.db[i] + momentum.db[i].normalized() * cache.cost * 0.025f;
		network->GetParams().W[i] -= learnRate * momentum.dW[i];
		network->GetParams().b[i] -= learnRate * momentum.db[i];
	}
}

void NetTrainer::UpdateParametersADAM() {
	float learnRate = ( trainParams.learningRate*trainParams.learningMod );
	for( int i = 0; i < (int)trainParams.dW.size(); ++i ) {
		NetTrainParameters vCorrected = momentum;
		NetTrainParameters sCorrected = momentumSqr;
		momentum.dW[i] = b1 * momentum.dW[i] + (invB1) * trainParams.dW[i];
		momentum.db[i] = b1 * momentum.db[i] + (invB1) * trainParams.db[i];
		vCorrected.dW[i] = momentum.dW[i] / ( invBSq1 );
		vCorrected.db[i] = momentum.db[i] / ( invBSq1 );
		momentumSqr.dW[i] = ( b2 * momentumSqr.dW[i] ) + ( (invB2) * MatrixXf(trainParams.dW[i].array() * trainParams.dW[i].array()) );
		momentumSqr.db[i] = ( b2 * momentumSqr.db[i] ) + ( (invB2) * MatrixXf(trainParams.db[i].array() * trainParams.db[i].array()) );
		sCorrected.dW[i] = momentumSqr.dW[i] / ( invBSq2 );
		sCorrected.db[i] = momentumSqr.db[i] / ( invBSq2 );
		network->GetParams().W[i] -= learnRate * MatrixXf(vCorrected.dW[i].array() / ( sCorrected.dW[i].array().sqrt() + FLT_EPSILON ));
		network->GetParams().b[i] -= learnRate * MatrixXf(vCorrected.db[i].array() / ( sCorrected.db[i].array().sqrt() + FLT_EPSILON ));
	}
}

void NetTrainer::BuildDropoutMask() {
	dropParams = network->GetParams();
	for(int i = 0; i < (int)network->GetParams().W.size() - 1; ++i) {
		for(int row = 0; row < network->GetParams().W[i].rows(); ++row) {
			float val = ((float)rand() / (RAND_MAX));
			if(val > 1.f) {
				dropParams.W[i].row(row) = MatrixXf::Zero(1, network->GetParams().W[i].cols());
				dropParams.b[i].row(row) = MatrixXf::Zero(1, network->GetParams().b[i].cols());
			}
		}
	}
}
void NetTrainer::UpdateSingleStep(){
	//BuildDropoutMask();
	cache.cost = CalcCost(&ForwardTrain(), trainLabels);
	BackwardPropagation();
	UpdateParametersADAM();
	cache.cost = CalcCost(cache.A.back(), *trainLabels);
}

