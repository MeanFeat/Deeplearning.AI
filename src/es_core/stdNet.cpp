#include "stdNet.h"
#include "json.h"

Net::Net(){
}

Net::Net(int inputSize, const vector<int> hiddenSizes, int outputSize, const vector<Activation> activations) {
	params.layerActivations = activations;
	params.layerSizes.push_back(inputSize);
	for(int l = 0; l < (int)hiddenSizes.size(); ++l) {
		params.layerSizes.push_back(hiddenSizes[l]);
	}
	params.layerSizes.push_back(outputSize);
	AddLayer(hiddenSizes[0], inputSize);
	for(int h = 1; h < (int)hiddenSizes.size(); ++h) {
		AddLayer(hiddenSizes[h], hiddenSizes[h - 1]);
	}
	AddLayer(outputSize, hiddenSizes.back());
}

Net::Net(const string fileName) {
	LoadNetwork(fileName);
}

Net::~Net() {
}

NetParameters &Net::GetParams() {
	return params;
}

void Net::SetParams(vector<MatrixXf> W, vector<MatrixXf> b) {
	params.W = W;
	params.b = b;
}

void Net::AddLayer(int A, int B) {
	params.W.push_back(MatrixXf::Zero(A, B));
	params.b.push_back(VectorXf::Zero(A, 1));
}

MatrixXf Net::Activate(Activation act, const MatrixXf &In) {
	switch( act ) {
		case Linear:
		return In;
		break;
		case Sigmoid:
		return CalcSigmoid(In);
		break;
		case Tanh:
		return CalcTanh(In);
		break;
		case ReLU:
		return CalcReLU(In);
		break;
		case LReLU:
		return CalcLReLU(In);
		break;
		case Sine:
		return CalcSine(In);
		break;
		default:
		return In;
		break;
	}
}

MatrixXf Net::ForwardPropagation(const MatrixXf &X) {
	MatrixXf lastOutput = X;
	for(int i = 0; i < (int)params.layerSizes.size() - 1; ++i) {
		MatrixXf weighed = params.W[i] * lastOutput;
		weighed.colwise() += (VectorXf)params.b[i];
		lastOutput = Activate(params.layerActivations[i], weighed);
	}
	return lastOutput;
}

void Net::SaveNetwork() {

}

void Net::LoadNetwork(const string fileName) {
	std::string line;
	ifstream file(fileName);
	vector<int> shapeDims;
	MatrixXf tempMat;
	ParseState state = ParseState::none;
	int element = 0, layerIndex = 0;
	while (file.good()) {
		std::getline(file, line);
		std::stringstream iss(line);
		std::string val, lastVal;
		while (iss.good()) {
			std::getline(iss, val, ':');
			if (state == ParseState::none) {
				if (strFind(lastVal, "layer")) {
					state = ParseState::layer;
					strCast(&layerIndex, val);
				}
				else if (strFind(lastVal, "activation")) {
					state = ParseState::activation;
					params.layerActivations.push_back(ReadActivation(val));
				}
				if (strFind(lastVal, "Shape")) {
					shapeDims.clear();
					state = strFind(lastVal, "weight") ? ParseState::weightShape : ParseState::biasShape;
				}
				else if (strFind(lastVal, "Vals")) {
					state = strFind(lastVal, "weight") ? ParseState::weightValues : ParseState::biasValues;
				}
				lastVal = val;
			}
			switch (state) {
			case none:
			case layer:
			case activation: //fall through
				state = ParseState::none;
				break;
			case weightShape:
			case biasShape:	//fall through
			{
				if (!strFind(val, "]")) {
					std::getline(iss, val, ',');
					if (!strFind(val, "[")) {
						int temp;
						strCast(&temp, val);
						shapeDims.push_back(temp);
						assert(shapeDims.size() <= 2);
					}
				}
				else {
					if (state == ParseState::weightShape) {
						if (layerIndex == 0) {
							params.layerSizes.push_back(shapeDims[0]);
						}
						params.layerSizes.push_back(shapeDims[1]);
					}
					tempMat = MatrixXf(shapeDims[0], shapeDims[1]);
					state = ParseState::none;
				}
			}
			break;
			case weightValues:
			case biasValues://fall through
			{
				if (!strFind(val, "]")) {
					std::getline(iss, val, ',');
					if (!strFind(val, "[")) {
						float temp;
						strCast(&temp, val);
						if (shapeDims[0] != 1 && shapeDims[1] != 1) {
							int r = int(floor(element % shapeDims[1]));
							int c = int(floor(element / shapeDims[1]));
							*(tempMat.data() + ((r * shapeDims[0]) + c)) = temp;
							element++;
						}
						else {
							*(tempMat.data() + element++) = temp;
						}
					}
				}
				if (element >= tempMat.size()) {
					element = 0;
					if (state == weightValues) {
						params.W.push_back(tempMat.transpose());
					}
					else {
						params.b.push_back(tempMat);
					}
					state = ParseState::none;
				}
			}
			break;
			default:
				state = ParseState::none;
				break;
			}
		}
	}
	file.close();
}


