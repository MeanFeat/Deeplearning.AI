#include "stdNet.h"

Net::Net() {
	params.learningRate = 0.01f;
}
Net::~Net() {}

NetParameters Net::GetParams() {
	return params;
}

NetCache Net::GetCache() {
	return cache;
}

void Net::InitializeParameters(int inputSize, int hiddenSize, int outputSize) {
	params.layerSizes.push_back(inputSize);
	params.layerSizes.push_back(hiddenSize);
	params.layerSizes.push_back(outputSize);
	params.W1 = MatrixXf::Random(hiddenSize, inputSize) * 0.01f;
	params.b1 = VectorXf::Zero(hiddenSize, 1);
	params.W2 = MatrixXf::Random(outputSize, hiddenSize) * 0.01f;
	params.b2 = VectorXf::Zero(outputSize, 1);
	return ;
}

MatrixXf Net::ForwardPropagation(MatrixXf X, bool training) {
	MatrixXf broadB1 = params.b1;
	MatrixXf broadB2 = params.b2;
	broadB1.conservativeResize(Eigen::NoChange, X.cols());
	broadB2.conservativeResize(Eigen::NoChange, X.cols());
	MatrixXf Z1 = (params.W1 * X) + broadB1;
	MatrixXf A1 = Tanh(Z1);
	MatrixXf Z2 = (params.W2 * A1) + broadB2;
	MatrixXf A2 = Sigmoid(Z2);
	if(training) {
		cache.Z1 = Z1;
		cache.A1 = A1;
		cache.Z2 = Z2;
		cache.A2 = A2;
	}
	return A2;
}

float Net::ComputeCost(MatrixXf A2, MatrixXf Y) {
	int m = (int)Y.cols();
	float coeff = 1.0f / m;
	return -((Y.cwiseProduct(Log(A2))) + (MatrixXf::Ones(1, m) - Y).cwiseProduct((Log(MatrixXf::Ones(1, m) - A2)))).sum() * coeff;
}

void Net::BackwardPropagation(MatrixXf X, MatrixXf Y) {
	int m = (int)Y.cols();
	float coeff = float(1.f / m);
	MatrixXf dZ2 = cache.A2 - Y;
	grads.dW2 = coeff * (dZ2* cache.A1.transpose());
	grads.db2 = coeff * dZ2.rowwise().sum();
	MatrixXf A1Squared = cache.A1.array().pow(2);
	MatrixXf Step2 = params.W2.transpose() * dZ2;
	MatrixXf Step3 = MatrixXf::Ones(cache.A1.cols(), cache.A1.rows()) - A1Squared.transpose();
	MatrixXf dZ1 = (Step2).cwiseProduct(Step3.transpose());
	MatrixXf Xprime = X.transpose();
	grads.dW1 = (dZ1 * Xprime) * coeff;
	grads.db1 = coeff * dZ1.rowwise().sum();
}

void Net::UpdateParameters() {
	params.W1 = params.W1 - (params.learningRate * grads.dW1);
	params.b1 = params.b1 - (params.learningRate * grads.db1);
	params.W2 = params.W2 - (params.learningRate * grads.dW2);
	params.b2 = params.b2 - (params.learningRate * grads.db2);
}

void Net::UpdateSingleStep(MatrixXf X, MatrixXf Y) {
	ForwardPropagation(X, true);
	cache.cost = ComputeCost(cache.A2, Y);
	BackwardPropagation(X, Y);
	UpdateParameters();
}
