#include "es_test.h"
#include <conio.h>
#include <iostream>
#include <fstream>

static vector<string> headers;
static vector<string> prefixes;
static vector<string> functionNames;
static vector<vector<vector<int>>> arguments;

void ReadTestList(const string fName) {
	std::string line;
	ifstream file(fName);
	ParseState state = ParseState::none;
	vector<int> parseArgs;
	vector<vector<int>> currentFuncArgs;
	while (file.good()) {
		std::getline(file, line);
		std::stringstream iss(line);
		std::string val, lastVal;
		while (iss.good()) {
			std::getline(iss, val, ':');
			if (state == ParseState::none) {
				if (strFind(lastVal, "header")) {
					state = ParseState::header;
					headers.push_back(val);
				} else if (strFind(lastVal, "prefix")) {
					state = ParseState::prefix;
					prefixes.push_back(val);
				} else if (strFind(lastVal, "functionName")) {
					state = ParseState::functionName;
					functionNames.push_back(val);
				} 
				if (strFind(lastVal, "arguments")) {
					state = ParseState::args;
				}
				lastVal = val;
			}
			switch (state) {
			case ParseState::none:
			case ParseState::header:
			case ParseState::prefix:
			case ParseState::functionName: //fall through
				state = ParseState::none;
				break;
			case ParseState::args: {
				if (strFind(val, "}")) {
					state = ParseState::none;
					arguments.push_back(currentFuncArgs);
					currentFuncArgs.clear();
				} else{
					if (!strFind(val, "{")) {
						size_t pos = 0;
						std::string token;
						do {
							token = val.substr(0, pos);
							val.erase(0, pos + 1);
							int temp;
							strCast(&temp, val);
							parseArgs.push_back(temp);
						} while ((pos = val.find(',')) != std::string::npos);
						currentFuncArgs.push_back(parseArgs);
						parseArgs.clear();
					}
				}
			} break;
			default:
				state = ParseState::none;
				break;
			}
		}
	}
	file.close();
}
void CreateGeneratedUnit( const string fileName ){
	ofstream file(fileName.c_str());
	file.clear();
	file << "//GENERATED FILE" << endl;
	for( int fn = 0; fn < functionNames.size(); fn++){
		string className = headers[fn];
		className.erase(remove_if(className.begin(), className.end(), 
			[](unsigned char x) {return x == '\"'; }), className.end());
		file << "TEST_CLASS(" << className << ") { public:" << endl;
		for (int arg = 0; arg < arguments[fn].size(); arg++){
			string args;
			file << "NAME_RUN(" << prefixes[fn] << "_";
			for (int a = 0; a < arguments[fn][arg].size(); a++) {
				file << arguments[fn][arg][a];
				if (a < arguments[fn][arg].size()-1) {
					file<<"x";
				} else {
					file << ",";
				}
			}
			file << functionNames[fn] << "(";
			for (int a = 0; a < arguments[fn][arg].size(); a++) {
				file << arguments[fn][arg][a];
				if (a < arguments[fn][arg].size() - 1) {
					file << ",";
				}
			}
			file << "));" << endl;
		}
		file << "};" << endl;
	}
	file.close();
}
void CreateGeneratedCpp( const string fileName ){
	ofstream file(fileName.c_str());
	file.clear();
	file << "//GENERATED FILE" << endl;
	for( int fn = 0; fn < functionNames.size(); fn++){
		file << "PrintHeader(" << headers[fn] << ");" << endl;
		for (int arg = 0; arg < arguments[fn].size(); arg++){
			file << functionNames[fn] << "(";
			for (int a = 0; a < arguments[fn][arg].size(); a++) {
				file << arguments[fn][arg][a];
				if (a < arguments[fn][arg].size()-1) {
					file << ", ";
				} else {
					file << ");" << endl;
				}
			}
		}
	}
	file.close();
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

void RunAllTests(){
	initParallel();
	setNbThreads(4);
	verbosity = 2;
#ifndef TEST_LISTS
#define TEST_LISTS
#include "test_cpp.generated"
#endif
}
int main() {
	cout << "(T)est, (B)uild, or e(X)it: ";
	HANDLE handle = GetStdHandle(STD_INPUT_HANDLE);
	DWORD events;
	INPUT_RECORD buffer;
	while (1) {
		PeekConsoleInput(handle, &buffer, 1, &events);
		if (events > 0){
			ReadConsoleInput(handle, &buffer, 1, &events);
			WORD in = buffer.Event.KeyEvent.wVirtualKeyCode;
			if (in == 66) { //'b'
				cout << "Building File" << endl;
				ReadTestList("tests.list");
				CreateGeneratedCpp("test_cpp.generated");
				CreateGeneratedUnit("test_unit.generated");
				break;
			} else if (in == 84) { //'t'
				cout << "Running All Tests" << endl;
				RunAllTests();
				break;
			}
			else if (in == 88) { //'x'
				break;
			}
		}
	}
	return 0;
}