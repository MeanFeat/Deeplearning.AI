#include "stdNetTrainer.h"

NetTrainer::NetTrainer() {
}


NetTrainer::NetTrainer(Net *net, MatrixXf *data, MatrixXf *labels, float weightScale, float learnRate, float regTerm) {
	network = net;
	trainData = data;
	trainLabels = labels;
	int nodeCount = 0;
	for( int i = 0; i < (int)network->GetParams().layerSizes.size() - 1; ++i ) {
		nodeCount += network->GetParams().layerSizes[i];
		if( network->GetParams().W[i].sum() == 0.f ) {	 //Don't initialize if we already have weights
			MatrixXf *w = &network->GetParams().W[i];
			*w = MatrixXf::Random(w->rows(), w->cols()) * weightScale;
		}
	}
	trainParams.learningMod = 1.f / nodeCount;
	trainParams.learningRate = learnRate;
	trainParams.regTerm = regTerm;
	for(int h = 1; h < (int)network->GetParams().layerSizes.size(); ++h) {
		AddLayer((int)network->GetParams().layerSizes[h], (int)network->GetParams().layerSizes[h - 1], weightScale);
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

void NetTrainer::AddLayer(int A, int B, float weightScale) {
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
	MatrixXf lastOutput = *trainData;
	for(int i = 0; i < (int)network->GetParams().layerSizes.size() - 1; ++i) {
		cache.Z[i] = (network->GetParams().W[i] * lastOutput).colwise() + (VectorXf)network->GetParams().b[i];
		lastOutput = Net::Activate(network->GetParams().layerActivations[i], cache.Z[i]);
		cache.A[i] = lastOutput;
	}
	return lastOutput;
}

float NetTrainer::CalcCost(const MatrixXf h, MatrixXf Y) {
	float coeff = 1.f / Y.cols();
	float sumSqrW = 0.f;
	for(int w = 0; w < (int)network->GetParams().W.size() - 1; ++w) {
		sumSqrW += network->GetParams().W[w].array().pow(2).sum();
	}
	float regCost = 0.5f * float((trainParams.regTerm*trainParams.learningMod) * (sumSqrW / (2.f * (float)trainLabels->cols())));
	return ((Y - h).array().pow(2).sum() * coeff) + regCost;
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
void NetTrainer::BackLayer(MatrixXf &dZ, int layerIndex, float coeff, const MatrixXf *LowerA) {
	dZ = BackActivation(layerIndex, dZ); 
	float lambda = 0.5f * ( trainParams.regTerm*trainParams.learningMod );
	MatrixXf *h = &trainParams.dW[layerIndex];
	*h = dZ * LowerA->transpose();
	int i = 0;
	while( i < h->size() ) {
		*( h->data() + i ) = coeff * ( *( h->data() + i ) + ( lambda * *( network->GetParams().W[layerIndex].data() + i ) ) );
		i++;
	}
	MatrixXf *db = &trainParams.db[layerIndex];
	*db = MatrixXf(dZ.rows(), 1);
	int b = 0;
	while( b < dZ.rows() ) {
		float rowSum = 0.f;
		for( int r = 0; r < dZ.cols(); r++ ) {
			rowSum += *( dZ.data() + ( b + dZ.rows() * r) );
		}
		*( db->data() + b ) = coeff * rowSum;
		b++;
	}
}
void NetTrainer::BackwardPropagation() {
	float coeff = float(1.f / (float)trainLabels->cols());
	MatrixXf dZ = MatrixXf(cache.A.back() - *trainLabels);
	trainParams.dW.back() = coeff * ( dZ * cache.A[cache.A.size() - 2].transpose() );
	trainParams.db.back() = coeff * dZ.rowwise().sum();
	for( int l = (int)network->GetParams().layerActivations.size() - 2; l >= 1; --l ) {
		BackLayer(dZ, l, coeff, &cache.A[l - 1]);
	}
	BackLayer(dZ, 0, coeff, trainData);
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
#define BETA2 (1.f - FLT_EPSILON)
void NetTrainer::UpdateParametersADAM() {
	for(int i = 0; i < (int)trainParams.dW.size(); ++i) {
		NetTrainParameters vCorrected = momentum;
		NetTrainParameters sCorrected = momentumSqr;
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
void NetTrainer::BuildDropoutMask() {
	dropParams = dropParams;
	dropParams.W[0] = MatrixXf::Ones(dropParams.W[0].rows(), dropParams.W[0].cols());
	dropParams.b[0] = MatrixXf::Ones(dropParams.b[0].rows(), dropParams.b[0].cols());
	for(int i = 1; i < (int)dropParams.W.size() - 1; ++i) {
		for(int row = 0; row < dropParams.W[i].rows(); ++row) {
			float val = ((float)rand() / (RAND_MAX));
			if(val < 0.95) {
				dropParams.W[i].row(row) = MatrixXf::Ones(1, dropParams.W[i].cols());
				dropParams.b[i].row(row) = MatrixXf::Ones(1, dropParams.b[i].cols());
			} else {
				dropParams.W[i].row(row) = MatrixXf::Zero(1, dropParams.W[i].cols());
				dropParams.b[i].row(row) = MatrixXf::Zero(1, dropParams.b[i].cols());
			}
		}
	}
	dropParams.W[dropParams.W.size() - 1].row(0) = MatrixXf::Ones(1, dropParams.W[dropParams.W.size() - 1].cols());
	dropParams.b[dropParams.b.size() - 1].row(0) = MatrixXf::Ones(1, dropParams.b[dropParams.b.size() - 1].cols());
}
void NetTrainer::UpdateSingleStep() {
	//BuildDropoutMask();
	cache.cost = CalcCost(ForwardTrain(), *trainLabels);
	BackwardPropagation();
	UpdateParametersADAM();
}
