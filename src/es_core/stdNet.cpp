#include "stdNet.h"
#include "es_parser.h"

using namespace Eigen;
using namespace std;
Net::Net(){}
Net::Net(int inputSize, std::vector<int> hiddenSizes, int outputSize, vector<Activation> activations){
	
	params.layerActivations = activations;
	params.layerSizes.push_back(inputSize);
	for(int l = 0; l < (int)hiddenSizes.size(); ++l){
		params.layerSizes.push_back(hiddenSizes[l]);
	}
	params.layerSizes.push_back(outputSize);
	AddLayer(hiddenSizes[0], inputSize);
	for(int h = 1; h < (int)hiddenSizes.size(); ++h){
		AddLayer(hiddenSizes[h], hiddenSizes[h - 1]);
	}
	AddLayer(outputSize, hiddenSizes.back());
}

Net::Net(const string fName){
	LoadNetwork(fName);
}

Net::~Net(){}
NetParameters &Net::GetParams(){
	return params;
}
void Net::SetParams(vector<MatrixXf> W, vector<MatrixXf> b){
	params.W = W;
	params.b = b;
}
void Net::AddLayer(int A, int B){
	params.W.push_back(MatrixXf::Random(A, B));
	params.b.push_back(MatrixXf::Zero(A, 1));
}
MatrixXf Net::Activate(const MatrixXf &In, Activation act) {
	switch(act){
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
		lastOutput = Net::Activate(weighed, params.layerActivations[i]);
	}
	return lastOutput;
}
void Net::SaveNetwork(){}

Activation ReadActivation(string str) {
	if (strFind(str, "sigmoid")) {
		return Activation::Sigmoid;
	}
	else if (strFind(str, "tanh")) {
		return Activation::Tanh;
	}
	else if (strFind(str, "relu")) {
		return Activation::ReLU;
	}
	else if (strFind(str, "leaky_relu")) {
		return Activation::LReLU;
	}
	else {
		return Activation::Linear;
	}
}

enum class NetParseState {
			layer,
			activation,
			weightShape,
			biasShape,
			weightValues,
			biasValues,
			none
		};

void Net::LoadNetwork(const string fName){
	std::string line;
	ifstream file(fName);
	vector<int> shapeDims;
	MatrixXf tempMat;
	NetParseState state = NetParseState::none;
	int element = 0, layerIndex = 0;
	while (file.good()) {
		std::getline(file, line);
		std::stringstream iss(line);
		std::string val, lastVal;
		while (iss.good()) {
			std::getline(iss, val, ':');
			if (state == NetParseState::none) {
				if (strFind(lastVal, "layer")) {
					state = NetParseState::layer;
					strCast(&layerIndex, val);
				}
				else if (strFind(lastVal, "activation")) {
					state = NetParseState::activation;
					params.layerActivations.push_back(ReadActivation(val));
				}
				if (strFind(lastVal, "Shape")) {
					shapeDims.clear();
					state = strFind(lastVal, "weight") ? NetParseState::weightShape : NetParseState::biasShape;
				}
				else if (strFind(lastVal, "Vals")) {
					state = strFind(lastVal, "weight") ? NetParseState::weightValues : NetParseState::biasValues;
				}
				lastVal = val;
			}
			switch (state) {
			case NetParseState::none:
			case NetParseState::layer:
			case NetParseState::activation: //fall through
				state = NetParseState::none;
				break;
			case NetParseState::weightShape:
			case NetParseState::biasShape:	//fall through
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
					if (state == NetParseState::weightShape) {
						if (layerIndex == 0) {
							params.layerSizes.push_back(shapeDims[0]);
						}
						params.layerSizes.push_back(shapeDims[1]);
					}
					tempMat = MatrixXf(shapeDims[0], shapeDims[1]);
					state = NetParseState::none;
				}
			}
			break;
			case NetParseState::weightValues:
			case NetParseState::biasValues://fall through
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
					if (state == NetParseState::weightValues) {
						params.W.push_back(tempMat.transpose());
					}
					else {
						params.b.push_back(tempMat);
					}
					state = NetParseState::none;
				}
			}
			break;
			default:
				state = NetParseState::none;
				break;
			}
		}
	}
	file.close();
}