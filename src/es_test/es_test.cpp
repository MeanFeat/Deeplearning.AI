#include <iostream>
#include <Eigen/dense>
#include "d_Matrix.h"
#include "d_math.h"
using namespace std;
using namespace Eigen;

static int verbosity = 1;
static const float thresholdMultiplier = (FLT_EPSILON * 0.4f);

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
static cudaStream_t cuda_stream;
void PrintOutcome(float controlSum, float testSum, float diff, float threshold, bool passed) {
	if (verbosity > 1) {
		cout << "Eigen: " << controlSum << " Device: " << testSum << endl;
		cout << "Error " << diff << " : " << threshold << endl;
	}
	if (verbosity > 0) {
		cout << "======================================================>> ";
		if (passed) {
			cout << "PASS!" << endl;
		}
		else {
			cout << "fail... " << diff - threshold << endl;
		}
	}
}
bool GetOutcome(float controlSum, float testSum, float threshold) {
	float diff = abs(controlSum - testSum);
	bool passed = diff < threshold;
	PrintOutcome(controlSum, testSum, diff, threshold, passed);
	return passed;
}
bool testMultipy(int m, int n, int k) {
	float threshold = float((m + k) * n) * thresholdMultiplier;
	cout << "Testing Multiply " << m << "," << n << " * " << n << "," << k << endl;
	MatrixXf A = MatrixXf::Random(m, n);
	MatrixXf B = MatrixXf::Random(n, k);
	d_Matrix d_A = to_device(A);
	d_Matrix d_B = to_device(B);
	d_Matrix d_C = to_device(MatrixXf::Zero(m, k));
	d_mult(&d_C, &d_A, &d_B);
	MatrixXf C = MatrixXf(to_host(d_C));
	float controlSum = (A*B).sum();
	float testSum = C.sum();
	float diff = abs(controlSum - testSum);
	bool passed = diff < threshold;
	PrintOutcome(controlSum, testSum, diff, threshold, passed);
	return passed;
}
void TestMultiplies() {
	cout << "======================" << endl;
	cout << "||Testing Multiplies||" << endl;
	cout << "======================" << endl;
	//testMultipy(1234, 98765, 654);
	//testMultipy(4321, 9595, 9462);
	testMultipy(9999, 85, 11111);
	testMultipy(100, 100, 100);
	testMultipy(8, 6002, 2);
}
bool testTransposeRight(int m, int n, int k) {
	float threshold = float((m + k) * n) * thresholdMultiplier;
	cout << "Testing Multiply (" << m << "," << n << ") * (" << n << "," << k << ").transpose()" <<endl;
	MatrixXf A = MatrixXf::Random(m, n);
	MatrixXf B = MatrixXf::Random(k, n);
	d_Matrix d_A = to_device(A);
	d_Matrix d_B = to_device(B);
	d_Matrix d_C = to_device(MatrixXf::Zero(m, k));
	d_mult_rhsT(&d_C, &d_A, &d_B);
	MatrixXf C = MatrixXf(to_host(d_C));
	float controlSum = (A*B.transpose()).sum();
	float testSum = C.sum();
	return GetOutcome(controlSum, testSum, threshold);
}
void TestMultsTransposeRight() {
	cout << "====================================" << endl;
	cout << "||Testing Multiply Transpose Right||" << endl;
	cout << "====================================" << endl;
	testTransposeRight(1234, 98765, 654);
	testTransposeRight(4321, 9595, 9462);
	testTransposeRight(9999, 85, 11111);
	testTransposeRight(100, 100, 100);
	testTransposeRight(8, 6002, 2);
}
bool testTransposeLeft(int m, int n, int k) {
	float threshold = float((m + k) * n) * thresholdMultiplier;
	cout << "Testing Multiply (" << m << "," << n << ").transpose() * (" << n << "," << k << ")" << endl;
	MatrixXf A = MatrixXf::Random(n, m);
	MatrixXf B = MatrixXf::Random(n, k);
	d_Matrix d_A = to_device(A);
	d_Matrix d_B = to_device(B);
	d_Matrix d_C = to_device(MatrixXf::Zero(m, k));
	d_mult_lhsT(&d_C, &d_A, &d_B);
	MatrixXf C = MatrixXf(to_host(d_C));
	float controlSum = (A.transpose()*B).sum();
	float testSum = C.sum();
	float diff = abs(controlSum - testSum);
	bool passed = diff < threshold;
	PrintOutcome(controlSum, testSum, diff, threshold, passed);
	return passed;
}
void TestMultsTransposeLeft() {
	cout << "====================================" << endl;
	cout << "||Testing Multiply Transpose Left||" << endl;
	cout << "====================================" << endl;
	testTransposeLeft(1234, 98765, 654);
	testTransposeLeft(4321, 9595, 9462);
	testTransposeLeft(9999, 85, 11111);
	testTransposeLeft(100, 100, 100);
	testTransposeLeft(8, 6002, 2);
}
bool TestSum(int m, int k) {
	cout << "Testing Sum " << m << "," << k << endl;
	MatrixXf A = MatrixXf::Random(m, k);
	d_Matrix d_A = to_device(A);
	float* d_testSum;
	float testSum;
	float controlSum = A.sum();
	cudaMalloc((void**)&d_testSum, sizeof(float));
	d_sumMatrix(d_testSum, &d_A);
	cudaMemcpy(&testSum, d_testSum, sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(d_testSum);
	float diff = abs(controlSum - testSum);
	float threshold = (m * k) * thresholdMultiplier;
	bool passed = diff < threshold;
	PrintOutcome(controlSum, testSum, diff, threshold, passed);
	return passed;
}
void TestSums() {
	cout << "================" << endl;
	cout << "||Testing Sums||" << endl;
	cout << "================" << endl;
	TestSum(1000, 1000000);
	TestSum(10000, 10000);
	TestSum(1111, 1131);
	TestSum(5000, 1);
}

int main() {
	initParallel();
	setNbThreads(4);
	verbosity = 2;
	//TestMultiplies();
	//TestMultsTransposeRight();
	//TestMultsTransposeLeft();

	TestSums();
}