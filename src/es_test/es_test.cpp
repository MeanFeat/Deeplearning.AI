#include "es_test_pch.h"
#include "es_test.h"
using namespace std;
using namespace Eigen;
void PrintHeader(string testType) {
	if (verbosity > 0) {
		const int len = (int)strlen(testType.c_str());
		string border = "============";
		for (int i = 0; i < len; i++) {
			border += "=";
		}
		cout << border << endl;
		cout << "||Testing " << testType << "||" << endl;
		cout << border << endl;
	}
}
string GetOutcomeString(const float cSum, const float tSum, const float diff, const float thresh, const bool passed) {
	string out;
	if (verbosity >= 2) {
		out += "Eigen: " + to_string(cSum) + " Device: " + to_string(tSum) + "\n";
		out += "Error " + to_string(diff) + " : Threshold " + to_string(thresh) + "\n";
	}
	if (verbosity >= 1) {
		out += "======================================================>> ";
		if (passed) {
			out += "PASS!";
		}
		else {
			out += "fail... " + to_string(diff - thresh);
		}
	}
	return out;
}
testResult GetOutcome(const float cSum, const float tSum, const float thresh) {
	testResult result;
	const float diff = abs(cSum - tSum);
	result.passed = diff <= abs(thresh);
	result.message = GetOutcomeString(cSum, tSum, diff, thresh, result.passed);
	const int passCol = diff > 0.f ? 14 : 10;
	TEXTCOLOUR(cout << result.message << endl; , result.passed ? passCol : 12);
	return result;
}
testResult testMultipy(const int m, const int n, const int k) {
	cout << "Testing Multiply " << m << "," << n << " * " << n << "," << k << endl;
	const testData A = testData(m, n);
	const testData B = testData(n, k);
	d_Matrix d_C = d_Matrix(m, k);
	d_mult(&d_C, &A.device, &B.device);
	const float threshold = float((m + k) * n) * thresholdMultiplier;
	return GetOutcome((A.host*B.host).sum(), MatrixXf(d_NetTrainer::to_host(d_C)).sum(), threshold);
}
testResult testTransposeRight(const int m, const int n, const int k) {
	cout << "Testing Multiply (" << m << "," << n << ") * (" << n << "," << k << ").transpose()" << endl;
	const testData A = testData(m, n);
	testData B = testData(k, n);
	d_Matrix d_C = d_Matrix(m, k);
	d_mult_rhsT(&d_C, &A.device, &B.device);
	const float threshold = float((m + k) * n) * thresholdMultiplier;
	return GetOutcome((A.host*B.host.transpose()).sum(), d_NetTrainer::to_host(d_C).sum(), threshold);
}
testResult testTransposeLeft(const int m, const int n, const int k) {
	cout << "Testing Multiply (" << m << "," << n << ").transpose() * (" << n << "," << k << ")" << endl;
	testData A = testData(n, m);
	const testData B = testData(n, k);
	d_Matrix d_C = d_Matrix(m, k);
	d_mult_lhsT(&d_C, &A.device, &B.device);
	const float threshold = float((m + k) * n) * thresholdMultiplier;
	return GetOutcome((A.host.transpose()*B.host).sum(), d_NetTrainer::to_host(d_C).sum(), threshold);
}
testResult testSum(const int m, const int k) {
	cout << "Testing Sum " << m << "," << k << endl;
	const testData A = testData(m, k);
	float* d_testSum;
	float result;
	cudaMalloc(&d_testSum, sizeof(float));
	d_sumMatrix(d_testSum, &A.device);
	cudaMemcpyAsync(&result, d_testSum, sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(d_testSum);
	const float threshold = float(m * k) * thresholdMultiplier;
	return GetOutcome(A.host.sum(), result, threshold);
}
testResult testTranspose(const int m, const int k) {
	cout << "Testing Transpose " << m << "," << k << endl;
	testData A = testData(m, k);
	d_Matrix d_C = d_Matrix(k, m);
	d_transpose(&d_C, &A.device);
	MatrixXf control = A.host.transpose();
	MatrixXf result = d_NetTrainer::to_host(d_C);
	string elemList = "";
	bool passed = true;
	for (int i = 0; i < result.size(); i++) {
		const float con = *(control.data() + i);
		const float res = *(result.data() + i);
		if (abs(con - res) > FLT_EPSILON) {
			passed = false;
			elemList += to_string(i) + ", ";
		}
	}
	return testResult(passed, elemList);
}
testResult testMultScalar(const int m, const int k) {
	cout << "Testing Multiply Element (" << m << "," << k << ") * b" << endl;
	testData A = testData(m, k);
	const float r = float(rand()) / float(RAND_MAX);
	d_Matrix d_C = d_NetTrainer::to_device(MatrixXf::Zero(m, k));
	d_mult_scalar(&A.device, r);
	const float controlSum = MatrixXf(A.host * r).sum();
	const float threshold = controlSum * thresholdMultiplier;
	return GetOutcome(controlSum, MatrixXf(d_NetTrainer::to_host(A.device)).sum(), threshold);
}
testResult testAdd(const int m, const int k) {
	cout << "Testing Add " << m << "," << k << " (+) " << m << "," << k << endl;
	testData A = testData(m, k);
	testData B = testData(m, k);
	d_Matrix d_C = d_Matrix(m, k);
	d_add_elem(&d_C, A.device, B.device);
	const float controlSum = MatrixXf(A.host.array() + B.host.array()).sum();
	const float threshold = controlSum * thresholdMultiplier;
	return GetOutcome(controlSum, MatrixXf(d_NetTrainer::to_host(d_C)).sum(), threshold);
}
testResult testSubtract(const int m, const int k) {
	cout << "Testing Subtract " << m << "," << k << " (-) " << m << "," << k << endl;
	testData A = testData(m, k);
	testData B = testData(m, k);
	d_Matrix d_C = d_Matrix(m, k);
	d_subtract_elem(&d_C, A.device, B.device);
	const float threshold = float(m + k) * thresholdMultiplier;
	return GetOutcome(MatrixXf(A.host.array() - B.host.array()).sum(), MatrixXf(d_NetTrainer::to_host(d_C)).sum(), threshold);
}
testResult testMultElem(const int m, const int k) {
	cout << "Testing MultElem " << m << "," << k << " (*) " << m << "," << k << endl;
	testData A = testData(m, k);
	testData B = testData(m, k);
	d_Matrix d_C = d_NetTrainer::to_device(MatrixXf::Zero(m, k));
	d_mult_elem(&d_C, A.device, B.device);
	const float threshold = float(m + k) * thresholdMultiplier;
	return GetOutcome(MatrixXf(A.host.array() * B.host.array()).sum(), MatrixXf(d_NetTrainer::to_host(d_C)).sum(), threshold);
}
testResult testSumRows(const int m, const int k) {
	cout << "Testing SumRows " << m << "," << k << endl;
	testData A = testData(m, k);
	d_Matrix d_C = d_NetTrainer::to_device(MatrixXf::Zero(m, 1));
	d_sumRows(&d_C, &A.device);
	MatrixXf result = MatrixXf(d_NetTrainer::to_host(d_C));
	MatrixXf control = A.host.rowwise().sum();
	string elemList = "";
	for (int i = 0; i < result.size(); i++) {
		const float r = *(result.data() + i);
		const float c = *(control.data() + i);
		elemList += to_string(r - c) + ",";
	}
	return testResult(result.isApprox(A.host.rowwise().sum()), elemList);
}
testResult testSet(const int m, const int k, const float val) {
	cout << "Testing Set " << m << "," << k << " (=) " << val << endl;
	const testData A = testData(m, k);
	d_Matrix d_C = A.device;
	d_set_elem(&d_C, val);
	MatrixXf result = d_NetTrainer::to_host(d_C);
	string elemList = "";
	bool passed = true;
	for (int i = 0; i < result.size(); i++) {
		const float ith = *(result.data() + i);
		if (ith != val) {
			passed = false;
			elemList += to_string(i) + ", ";
		}
	}
	return testResult(passed, elemList);
}
testResult testSquare(const int m, const int k) {
	cout << "Testing Square " << m << "," << k << endl;
	testData A = testData(m, k);
	d_Matrix d_C = d_Matrix(k, m);
	d_square(&d_C, &A.device);
	const float controlSum = MatrixXf(A.host.array()* A.host.array()).sum();
	const float threshold = controlSum * 0.00001f;
	return GetOutcome(controlSum, MatrixXf(d_NetTrainer::to_host(d_C)).sum(), threshold);
}
testResult testSigmoid(const int m, const int k) {
	cout << "Testing Sigmoid " << m << "," << k << endl;
	testData A = testData(m, k);
	d_activate(&A.device, Activation::Sigmoid);
	const float controlSum = MatrixXf(stdNet::Activate(A.host, Activation::Sigmoid)).sum();
	const float threshold = float(m * k) * thresholdMultiplier;
	return GetOutcome(controlSum, MatrixXf(d_NetTrainer::to_host(A.device)).sum(), threshold);
}
testResult testTanh(const int m, const int k) {
	cout << "Testing Tanh " << m << "," << k << endl;
	testData A = testData(m, k);
	d_activate(&A.device, Activation::Tanh);
	const float controlSum = MatrixXf(stdNet::Activate(A.host, Activation::Tanh)).sum();
	const float threshold = float(m * k) * thresholdMultiplier;
	return GetOutcome(controlSum, MatrixXf(d_NetTrainer::to_host(A.device)).sum(), threshold);
}
testResult testReLU(const int m, const int k) {
	cout << "Testing ReLU " << m << "," << k << endl;
	testData A = testData(m, k);
	d_activate(&A.device, Activation::ReLU);
	const float controlSum = MatrixXf(stdNet::Activate(A.host, Activation::ReLU)).sum();
	const float threshold = float(m * k) * thresholdMultiplier;
	return GetOutcome(controlSum, MatrixXf(d_NetTrainer::to_host(A.device)).sum(), threshold);
}
testResult testLReLU(const int m, const int k) {
	cout << "Testing ReLU " << m << "," << k << endl;
	testData A = testData(m, k);
	d_activate(&A.device, Activation::LReLU);
	const float controlSum = MatrixXf(stdNet::Activate(A.host, Activation::LReLU)).sum();
	const float threshold = float(m * k) * thresholdMultiplier;
	return GetOutcome(controlSum, MatrixXf(d_NetTrainer::to_host(A.device)).sum(), threshold);
}
testResult testSine(const int m, const int k) {
	cout << "Testing Sine " << m << "," << k << endl;
	testData A = testData(m, k);
	d_activate(&A.device, Activation::Sine);
	const float controlSum = MatrixXf(stdNet::Activate(A.host, Activation::Sine)).sum();
	const float threshold = float(m * k) * thresholdMultiplier;
	return GetOutcome(controlSum, MatrixXf(d_NetTrainer::to_host(A.device)).sum(), threshold);
}
testResult testBackProp(stdNet &nn, const int dataCount) {
	const MatrixXf data = MatrixXf::Random(nn.GetInputSize(), dataCount);
	const MatrixXf labels = MatrixXf::Random(nn.GetOutputSize(), dataCount);
	nn.RandomInit(0.15f);
	stdNet d_nn = (nn);
	NetTrainer h_trainer = NetTrainer(&nn, data, labels, 1.f, 0.25f, 0.f);
	h_trainer.ForwardTrain();
	h_trainer.BackwardPropagation();
	const MatrixXf control = h_trainer.GetTrainParams().dW[0];
	d_NetTrainer d_trainer = d_NetTrainer(&d_nn, data, labels, 1.f, 0.25f, 0.f);
	d_trainer.ForwardTrain();
	d_trainer.BackwardPropagation();
	const MatrixXf test = d_NetTrainer::to_host(d_trainer.GetDerivatives().d_dW[0]);
	return GetOutcome(control.sum(), test.sum(), dataCount * nn.GetNodeCount() * thresholdMultiplier);
}
testResult testCalcCost(stdNet &nn, const int dataCount) {
	const testData test = testData(nn.GetOutputSize(), dataCount);
	const testData labels = testData(nn.GetOutputSize(), dataCount);
	nn.RandomInit(0.15f);
	NetTrainer h_trainer = NetTrainer(&nn, test.host, labels.host, 1.f, 0.25f, 0.f);
	const float control = h_trainer.CalcCost(test.host, labels.host);
	d_NetTrainer d_trainer = d_NetTrainer(&nn, test.host, labels.host, 1.f, 0.25f, 0.f);
	const float result = d_trainer.CalcCost(test.device, labels.device);
	return GetOutcome(control, result, dataCount * nn.GetNodeCount() * thresholdMultiplier);
}
testResult testForward(stdNet &nn, const int dataCount) {
	const testData data = testData(nn.GetInputSize(), dataCount);
	const testData labels = testData(nn.GetOutputSize(), dataCount);
	nn.RandomInit(0.15f);
	NetTrainer h_trainer = NetTrainer(&nn, data.host, labels.host, 1.f, 0.25f, 0.f);
	const d_NetTrainer d_trainer = d_NetTrainer(&nn, data.host, labels.host, 1.f, 0.25f, 0.f);
	const MatrixXf control = nn.ForwardPropagation(data.host);
	const d_Matrix result = d_trainer.Forward(data.device);
	const MatrixXf test = d_NetTrainer::to_host(result);
	return GetOutcome(control.sum(), test.sum(), dataCount * nn.GetNodeCount() * thresholdMultiplier);
}
testResult testForwardTrain(stdNet &nn, const int dataCount) {
	const MatrixXf data = MatrixXf::Random(nn.GetInputSize(), dataCount);
	const MatrixXf labels = MatrixXf::Random(nn.GetOutputSize(), dataCount);
	nn.RandomInit(0.15f);
	NetTrainer h_trainer = NetTrainer(&nn, data, labels, 1.f, 0.25f, 0.f);
	const MatrixXf control = h_trainer.ForwardTrain();
	d_NetTrainer d_trainer = d_NetTrainer(&nn, data, labels, 1.f, 0.25f, 0.f);
	d_trainer.ForwardTrain();
	const MatrixXf test = d_NetTrainer::to_host(d_trainer.GetCache().d_A.back());
	return GetOutcome(control.sum(), test.sum(), dataCount * nn.GetNodeCount() * thresholdMultiplier);
}
