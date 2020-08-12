#include "es_test.h"
#include <iostream>
d_Matrix to_device(MatrixXf matrix) {
	//transpose data only to Column Major
	MatrixXf temp = matrix.transpose();
	return d_Matrix(temp.data(), (int)matrix.rows(), (int)matrix.cols());
}
MatrixXf to_host(d_Matrix d_matrix) {
	// return to Row Major order
	MatrixXf out = MatrixXf(d_matrix.cols(), d_matrix.rows());
	d_check(cudaMemcpy(out.data(), d_matrix.d_data(), d_matrix.memSize(), cudaMemcpyDeviceToHost));
	return out.transpose();
}
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
	if (passed)	{
		out += "PASS\n";
	} else {
		out += "Fail:\n";
	}
	if (verbosity > 1) {
		out += "Eigen: " + to_string(cSum) + " Device: " + to_string(tSum) + "\n";
		out += "Error " + to_string(diff) + " : " + to_string(thresh) + "\n";
	}
	if (verbosity > 0) {
		out += "======================================================>> ";
		if (passed) {
			out += "PASS!\n";
		}
		else {
			out += "fail... " + to_string(diff - thresh) + "\n";
		}
	}
	return out;
}
string GetOutcome(float cSum, float tSum, float thresh) {
	float diff = abs(cSum - tSum);
	bool passed = diff <= abs(thresh);
	string out = GetOutcomeString(cSum, tSum, diff, thresh, passed);
	cout << out;
	return out;
}
string testMultipy(int m, int n, int k) {
	cout << "Testing Multiply " << m << "," << n << " * " << n << "," << k << endl;
	MatrixXf A = MatrixXf::Random(m, n);
	MatrixXf B = MatrixXf::Random(n, k);
	d_Matrix d_A = to_device(A);
	d_Matrix d_B = to_device(B);
	d_Matrix d_C = to_device(MatrixXf::Zero(m, k));
	d_mult(&d_C, &d_A, &d_B);
	float threshold = float((m + k) * n) * thresholdMultiplier;
	return GetOutcome((A*B).sum(), MatrixXf(to_host(d_C)).sum(), threshold);
}
string testTransposeRight(int m, int n, int k) {
	cout << "Testing Multiply (" << m << "," << n << ") * (" << n << "," << k << ").transpose()" << endl;
	MatrixXf A = MatrixXf::Random(m, n);
	MatrixXf B = MatrixXf::Random(k, n);
	d_Matrix d_A = to_device(A);
	d_Matrix d_B = to_device(B);
	d_Matrix d_C = to_device(MatrixXf::Zero(m, k));
	d_mult_rhsT(&d_C, &d_A, &d_B);
	MatrixXf C = MatrixXf(to_host(d_C));
	float threshold = float((m + k) * n) * thresholdMultiplier;
	return GetOutcome((A*B.transpose()).sum(), C.sum(), threshold);
}
string testTransposeLeft(int m, int n, int k) {
	cout << "Testing Multiply (" << m << "," << n << ").transpose() * (" << n << "," << k << ")" << endl;
	MatrixXf A = MatrixXf::Random(n, m);
	MatrixXf B = MatrixXf::Random(n, k);
	d_Matrix d_A = to_device(A);
	d_Matrix d_B = to_device(B);
	d_Matrix d_C = to_device(MatrixXf::Zero(m, k));
	d_mult_lhsT(&d_C, &d_A, &d_B);
	MatrixXf C = MatrixXf(to_host(d_C));
	float threshold = float((m + k) * n) * thresholdMultiplier;
	return GetOutcome((A.transpose()*B).sum(), C.sum(), threshold);
}
string testSum(int m, int k) {
	cout << "Testing Sum " << m << "," << k << endl;
	MatrixXf A = MatrixXf::Random(m, k);
	d_Matrix d_A = to_device(A);
	float* d_testSum;
	float testSum;
	cudaMalloc((void**)&d_testSum, sizeof(float));
	d_sumMatrix(d_testSum, &d_A);
	cudaMemcpy(&testSum, d_testSum, sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(d_testSum);
	float threshold = float(m + k) * thresholdMultiplier;
	return GetOutcome(A.sum(), testSum, m * k * thresholdMultiplier);
}
string testTranspose(int m, int k) {
	cout << "Testing Transpose " << m << "," << k << endl;
	MatrixXf A = MatrixXf::Random(m, k);
	d_Matrix d_A = to_device(A);
	d_Matrix d_testTranspose = to_device(MatrixXf::Ones(k, m));
	MatrixXf controlTranspose = A.transpose();
	d_transpose(&d_testTranspose, &d_A);
	MatrixXf testTranspose = to_host(d_testTranspose);
	float threshold = float(m * k) * thresholdMultiplier;
	return GetOutcome(controlTranspose.sum(), testTranspose.sum(), threshold);
}
string testMultScalar(int m, int k) {
	cout << "Testing Multiply Element (" << m << "," << k << ") * b" << endl;
	MatrixXf A = MatrixXf::Random(m, k);
	float r = float(rand() / RAND_MAX);
	d_Matrix d_A = to_device(A);
	d_Matrix d_C = to_device(MatrixXf::Zero(m, k));
	d_mult_scalar(&d_A, r);
	float controlSum = MatrixXf(A * r).sum();
	float threshold = controlSum * thresholdMultiplier;
	return GetOutcome(controlSum, MatrixXf(to_host(d_A)).sum(), threshold);
}
string testAdd(int m, int k) {
	cout << "Testing Add " << m << "," << k << " (+) " << m << "," << k << endl;
	MatrixXf A = MatrixXf::Random(m, k);
	MatrixXf B = MatrixXf::Random(m, k);
	d_Matrix d_A = to_device(A);
	d_Matrix d_B = to_device(B);
	d_Matrix d_C = to_device(MatrixXf::Zero(m, k));
	d_add_elem(&d_C, &d_A, &d_B);
	float controlSum = MatrixXf(to_host(d_C)).sum();
	float threshold = controlSum * thresholdMultiplier;
	return GetOutcome(controlSum, MatrixXf(to_host(d_C)).sum(), threshold);
}
string testSubtract(int m, int k) {
	cout << "Testing Add " << m << "," << k << " (-) " << m << "," << k << endl;
	MatrixXf A = MatrixXf::Random(m, k);
	MatrixXf B = MatrixXf::Random(m, k);
	d_Matrix d_A = to_device(A);
	d_Matrix d_B = to_device(B);
	d_Matrix d_C = to_device(MatrixXf::Zero(m, k));
	d_subtract_elem(&d_C, &d_A, &d_B);
	float threshold = float(m + k) * thresholdMultiplier;
	return GetOutcome(MatrixXf(A.array() - B.array()).sum(), MatrixXf(to_host(d_C)).sum(), threshold);
}
string testSquare(int m, int k) {
	cout << "Testing Square " << m << "," << k << endl;
	MatrixXf A = MatrixXf::Random(m, k);
	d_Matrix d_A = to_device(A);
	d_Matrix d_C = to_device(MatrixXf::Zero(k, m));
	d_square(&d_C, &d_A);
	float controlSum = MatrixXf(A.array()* A.array()).sum();
	float threshold = controlSum * 0.00001f;
	return GetOutcome(controlSum, MatrixXf(to_host(d_C)).sum(), threshold);
}
string testSigmoid(int m, int k) {
	cout << "Testing Sigmoid " << m << "," << k << endl;
	MatrixXf A = MatrixXf::Random(m, k);
	d_Matrix d_A = to_device(A);
	d_activate(&d_A, Activation::Sigmoid);
	float controlSum = MatrixXf(Net::Activate(Activation::Sigmoid, A)).sum();
	float threshold = controlSum * thresholdMultiplier;
	return GetOutcome(controlSum, MatrixXf(to_host(d_A)).sum(), threshold);
}
string testTanh(int m, int k) {
	cout << "Testing Tanh " << m << "," << k << endl;
	MatrixXf A = MatrixXf::Random(m, k);
	d_Matrix d_A = to_device(A);
	d_activate(&d_A, Activation::Tanh);
	float controlSum = MatrixXf(Net::Activate(Activation::Tanh, A)).sum();
	float threshold = controlSum * thresholdMultiplier;
	return GetOutcome(controlSum, MatrixXf(to_host(d_A)).sum(), threshold);
}
string testReLU(int m, int k) {
	cout << "Testing ReLU " << m << "," << k << endl;
	MatrixXf A = MatrixXf::Random(m, k);
	d_Matrix d_A = to_device(A);
	d_activate(&d_A, Activation::ReLU);
	float controlSum = MatrixXf(Net::Activate(Activation::ReLU, A)).sum();
	float threshold = controlSum * thresholdMultiplier;
	return GetOutcome(controlSum, MatrixXf(to_host(d_A)).sum(), threshold);
}
string testLReLU(int m, int k) {
	cout << "Testing ReLU " << m << "," << k << endl;
	MatrixXf A = MatrixXf::Random(m, k);
	d_Matrix d_A = to_device(A);
	d_activate(&d_A, Activation::LReLU);
	float controlSum = MatrixXf(Net::Activate(Activation::LReLU, A)).sum();
	float threshold = controlSum * thresholdMultiplier;
	return GetOutcome(controlSum, MatrixXf(to_host(d_A)).sum(), threshold);
}
