#include "es_test_pch.h"
#include "es_test.h"
using namespace std;
using namespace Eigen;
void PrintHeader(string testType) {
	if (verbosity > 0) {
		int len = (int)strlen(testType.c_str());
		string border = "============";
		for (int i = 0; i < len; i++) {
			border += "=";
		}
		cout << border << endl;
		cout << "||Testing " << testType << "||" << endl;
		cout << border << endl;
	}
}
string GetOutcomeString(float cSum, float tSum, float diff, float thresh, bool passed) {
	string out;
	if (verbosity >= 2) {
		out += "Eigen: " + to_string(cSum) + " Device: " + to_string(tSum) + "\n";
		out += "Error " + to_string(diff) + " : " + to_string(thresh) + "\n";
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
testResult GetOutcome(float cSum, float tSum, float thresh) {
	testResult result;
	float diff = abs(cSum - tSum);
	result.passed = diff <= abs(thresh);
	result.message = GetOutcomeString(cSum, tSum, diff, thresh, result.passed);
	int passCol = diff > 0.f ? 14 : 10;
	TEXTCOLOUR(cout << result.message << endl;, result.passed ? passCol : 12);
	return result;
}
testResult testMultipy(int m, int n, int k) {
	cout << "Testing Multiply " << m << "," << n << " * " << n << "," << k << endl;
	testData A = testData(m, n);
	testData B = testData(n, k);
	d_Matrix d_C = d_Matrix(m, k);
	d_mult(&d_C, &A.device, &B.device);
	float threshold = float((m + k) * n) * thresholdMultiplier;
	return GetOutcome((A.host*B.host).sum(), MatrixXf(to_host(d_C)).sum(), threshold);
}
testResult testTransposeRight(int m, int n, int k) {
	cout << "Testing Multiply (" << m << "," << n << ") * (" << n << "," << k << ").transpose()" << endl;
	testData A = testData(m, n);
	testData B = testData(k, n);
	d_Matrix d_C = d_Matrix(m, k);
	d_mult_rhsT(&d_C, &A.device, &B.device);
	float threshold = float((m + k) * n) * thresholdMultiplier;
	return GetOutcome((A.host*B.host.transpose()).sum(), to_host(d_C).sum(), threshold);
}
testResult testTransposeLeft(int m, int n, int k) {
	cout << "Testing Multiply (" << m << "," << n << ").transpose() * (" << n << "," << k << ")" << endl;
	testData A = testData(n, m);
	testData B = testData(n, k);
	d_Matrix d_C = d_Matrix(m, k);
	d_mult_lhsT(&d_C, &A.device, &B.device);
	float threshold = float((m + k) * n) * thresholdMultiplier;
	return GetOutcome((A.host.transpose()*B.host).sum(), to_host(d_C).sum(), threshold);
}
testResult testSum(int m, int k) {
	cout << "Testing Sum " << m << "," << k << endl;
	testData A = testData(m, k);
	float* d_testSum;
	float result;
	cudaMalloc((void**)&d_testSum, sizeof(float));
	d_sumMatrix(d_testSum, &A.device);
	cudaMemcpy(&result, d_testSum, sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(d_testSum);
	float threshold = float(m + k) * thresholdMultiplier;
	return GetOutcome(A.host.sum(), result, m * k * thresholdMultiplier);
}
testResult testTranspose(int m, int k) {
	cout << "Testing Transpose " << m << "," << k << endl;
	testData A = testData(m, k);
	d_Matrix d_C = d_Matrix(k, m);
	d_transpose(&d_C, &A.device);
	MatrixXf control = A.host.transpose();
	MatrixXf result = to_host(d_C);
	string elemList = "";
	bool passed = true;
	for (int i = 0; i < result.size(); i++) {
		float con = *(control.data() + i);
		float res = *(result.data() + i);
		if (abs(con - res) > FLT_EPSILON) {
			passed = false;
			elemList += to_string(i) + ", ";
		}
	}
	return testResult(passed, elemList);
}
testResult testMultScalar(int m, int k) {
	cout << "Testing Multiply Element (" << m << "," << k << ") * b" << endl;
	testData A = testData(m, k);
	float r = float(rand() / RAND_MAX);
	d_Matrix d_C = to_device(MatrixXf::Zero(m, k));
	d_mult_scalar(&A.device, r);
	float controlSum = MatrixXf(A.host * r).sum();
	float threshold = controlSum * thresholdMultiplier;
	return GetOutcome(controlSum, MatrixXf(to_host(A.device)).sum(), threshold);
}
testResult testAdd(int m, int k) {
	cout << "Testing Add " << m << "," << k << " (+) " << m << "," << k << endl;
	testData A = testData(m, k);
	testData B = testData(m, k);
	d_Matrix d_C = d_Matrix(m, k);
	d_add_elem(&d_C, A.device, B.device);
	float controlSum = MatrixXf(A.host.array() + B.host.array()).sum();
	float threshold = controlSum * thresholdMultiplier;
	return GetOutcome(controlSum, MatrixXf(to_host(d_C)).sum(), threshold);
}
testResult testSubtract(int m, int k) {
	cout << "Testing Subtract " << m << "," << k << " (-) " << m << "," << k << endl;
	testData A = testData(m, k);
	testData B = testData(m, k);
	d_Matrix d_C = d_Matrix(m, k);
	d_subtract_elem(&d_C, A.device, B.device);
	float threshold = float(m + k) * thresholdMultiplier;
	return GetOutcome(MatrixXf(A.host.array() - B.host.array()).sum(), MatrixXf(to_host(d_C)).sum(), threshold);
}
testResult testMultElem(int m, int k) {
	cout << "Testing MultElem " << m << "," << k << " (*) " << m << "," << k << endl;
	testData A = testData(m, k);
	testData B = testData(m, k);
	d_Matrix d_C = to_device(MatrixXf::Zero(m, k));
	d_mult_elem(&d_C, A.device, B.device);
	float threshold = float(m + k) * thresholdMultiplier;
	return GetOutcome(MatrixXf(A.host.array() * B.host.array()).sum(), MatrixXf(to_host(d_C)).sum(), threshold);
}
testResult testSet(int m, int k, float val) {
	cout << "Testing Set " << m << "," << k << " (=) " << val << endl;
	testData A = testData(m, k);
	d_Matrix d_C = A.device;
	d_set_elem(&d_C, val);
	MatrixXf result = to_host(d_C);
	string elemList = "";
	bool passed = true;
	for (int i = 0; i < result.size(); i++) {
		float ith = *(result.data() + i);
		if (ith != val) {
			passed = false;
			elemList += to_string(i) + ", ";
		}
	}
	return testResult(passed, elemList);
}
testResult testSquare(int m, int k) {
	cout << "Testing Square " << m << "," << k << endl;
	testData A = testData(m, k);
	d_Matrix d_C = d_Matrix(k, m);
	d_square(&d_C, &A.device);
	float controlSum = MatrixXf(A.host.array()* A.host.array()).sum();
	float threshold = controlSum * 0.00001f;
	return GetOutcome(controlSum, MatrixXf(to_host(d_C)).sum(), threshold);
}
testResult testSigmoid(int m, int k) {
	cout << "Testing Sigmoid " << m << "," << k << endl;
	testData A = testData(m, k);
	d_activate(&A.device, Activation::Sigmoid);
	float controlSum = MatrixXf(Net::Activate(A.host, Activation::Sigmoid)).sum();
	float threshold = m + k * thresholdMultiplier;
	return GetOutcome(controlSum, MatrixXf(to_host(A.device)).sum(), threshold);
}
testResult testTanh(int m, int k) {
	cout << "Testing Tanh " << m << "," << k << endl;
	testData A = testData(m, k);
	d_activate(&A.device, Activation::Tanh);
	float controlSum = MatrixXf(Net::Activate(A.host, Activation::Tanh)).sum();
	float threshold = m + k * thresholdMultiplier;
	return GetOutcome(controlSum, MatrixXf(to_host(A.device)).sum(), threshold);
}
testResult testReLU(int m, int k) {
	cout << "Testing ReLU " << m << "," << k << endl;
	testData A = testData(m, k);
	d_activate(&A.device, Activation::ReLU);
	float controlSum = MatrixXf(Net::Activate(A.host, Activation::ReLU)).sum();
	float threshold = controlSum * thresholdMultiplier;
	return GetOutcome(controlSum, MatrixXf(to_host(A.device)).sum(), threshold);
}
testResult testLReLU(int m, int k) {
	cout << "Testing ReLU " << m << "," << k << endl;
	testData A = testData(m, k);
	d_activate(&A.device, Activation::LReLU);
	float controlSum = MatrixXf(Net::Activate(A.host, Activation::LReLU)).sum();
	float threshold = controlSum * thresholdMultiplier;
	return GetOutcome(controlSum, MatrixXf(to_host(A.device)).sum(), threshold);
}