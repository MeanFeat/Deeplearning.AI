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
void PrintOutcome(float cSum, float tSum, float diff, float thresh, bool passed) {
	if (verbosity > 1) {
		cout << "Eigen: " << cSum << " Device: " << tSum << endl;
		cout << "Error " << diff << " : " << thresh << endl;
	}
	if (verbosity > 0) {
		cout << "======================================================>> ";
		if (passed) {
			cout << "PASS!" << endl;
		}
		else {
			cout << "fail... " << diff - thresh << endl;
		}
	}
}
bool GetOutcome(float cSum, float tSum, float thresh) {
	float diff = abs(cSum - tSum);
	bool passed = diff < thresh;
	PrintOutcome(cSum, tSum, diff, thresh, passed);
	return passed;
}
bool testMultipy(int m, int n, int k) {
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
bool testTransposeRight(int m, int n, int k) {
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
bool testTransposeLeft(int m, int n, int k) {
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
bool testSum(int m, int k) {
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
bool testTranspose(int m, int k) {
	cout << "Testing Transpose " << m << "," << k << endl;
	MatrixXf A = MatrixXf::Random(m, k);
	d_Matrix d_A = to_device(A);
	d_Matrix d_testTranspose = to_device(MatrixXf::Ones(k, m));
	MatrixXf controlTranspose = A.transpose();
	d_transpose(&d_testTranspose, &d_A);
	MatrixXf testTranspose = to_host(d_testTranspose);
	float threshold = float(m + k) * thresholdMultiplier;
	return GetOutcome(controlTranspose.sum(), testTranspose.sum(), threshold);
}
