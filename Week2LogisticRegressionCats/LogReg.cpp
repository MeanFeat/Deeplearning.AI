
#include "stdafx.h"
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <unsupported/Eigen/MatrixFunctions>
#include <assert.h>

struct LogRegSet {
	MatrixXf w;
	float b;
};

struct LRTrainingSet {
	MatrixXf dw;
	float db;
	float Cost;
};

LogRegSet InitToZero(int dim) {
	LogRegSet Result;
	Result.w = MatrixXf::Zero(dim, 1);
	Result.b = 0.f;
	return Result;
}

MatrixXf GetModelOutput(LogRegSet set, MatrixXf X) {
	MatrixXf biases = MatrixXf::Ones(1, X.cols()) * set.b;
	return Sigmoid(set.w.transpose() * X + biases);
}

LRTrainingSet propegate(LogRegSet set, MatrixXf X, MatrixXf Y) {
	LRTrainingSet Result;
	int m = (int)Y.cols();
	float coeff = 1.f/ m;
	MatrixXf A = GetModelOutput(set, X);
	Result.Cost = -((Y.cwiseProduct(Log(A))) + (MatrixXf::Ones(1, m) - Y).cwiseProduct( (Log(MatrixXf::Ones(1, m) - A)))).sum() * coeff;
	Result.dw = (X * (A - Y).transpose()) * coeff;
	Result.db = (A - Y).sum()*coeff;
	return Result;
}


MatrixXf predict(LogRegSet lrs, MatrixXf X) {
	return Round(GetModelOutput(lrs, X));
}

void optimize(LogRegSet *lrsPtr, LRTrainingSet *trainSetPtr, MatrixXf X, MatrixXf Y, int iterations, float learnRate) {
	for(int i = 0; i < iterations; i++) {
		*trainSetPtr = propegate(*lrsPtr, X, Y);
		lrsPtr->w = lrsPtr->w - (learnRate * trainSetPtr->dw);
		lrsPtr->b = lrsPtr->b - (learnRate * trainSetPtr->db);
		Assert(trainSetPtr->dw.rows() == lrsPtr->w.rows() && trainSetPtr->dw.cols() == lrsPtr->w.cols());
		if(i % 100 == 0) {
			cout << "Cost: " << (float)trainSetPtr->Cost << " //==// train accuracy: " << 100 - (Abs(predict(*lrsPtr, X) - Y).sum() / Y.cols() * 100) << "%\n";
		}
	}
}

MatrixXf BuildMatFromFile(string fName, MatrixXf mat) {
	cout << "Loading File : " << fName << endl;
	MatrixXf tempMat = mat;
	ifstream file(fName);
	for(int row = 0; row < mat.rows(); ++row) {
		std::string line;
		std::getline(file, line);
		if(!file.good())
			break;
		std::stringstream iss(line);
		for(int col = 0; col < mat.cols(); ++col) {
			std::string val;
			std::getline(iss, val, ',');
			if(!iss.good())
				break;
			std::stringstream convertor(val);
			convertor>> tempMat(row,col);
		}
		if(row % 50 == 0) {
			cout << "Percent: " << (float)row / (float)mat.rows() * 100.f << "%... \r";
		}
	}
	cout << "Percent: 100.0% \r";
	cout << endl;
	return tempMat;
}

void PrintBool(bool inp) {
	if(inp) {
		cout << "             00 " << endl;
		cout << "            00  " << endl;
		cout << "           00   " << endl;
		cout << "00       000    " << endl;
		cout << "  00    00      " << endl;
		cout << "    00 00       " << endl;
		cout << "     000        " << endl;
	} else {
		cout << "  XX     XX       " << endl;
		cout << "   XX   XX        " << endl;
		cout << "    XX XX         " << endl;
		cout << "     XXX           " << endl;
		cout << "    XX XX          " << endl;
		cout << "   XX   XX         " << endl;
		cout << "  XX     XX        " << endl;
	}
}

int main() {
	LogRegSet lrs = InitToZero(12288);
	MatrixXf train_set_x = BuildMatFromFile("catTrain.csv", MatrixXf::Zero(12288, 209));
	MatrixXf test_set_x = BuildMatFromFile("catTest.csv", MatrixXf::Zero(12288, 50));
	MatrixXf train_set_y = BuildMatFromFile("catTrainLabels.csv", MatrixXf::Zero(1, 209));
	MatrixXf test_set_y = BuildMatFromFile("catTestLabels.csv", MatrixXf::Zero(1, 50));
	Assert(train_set_x.rows() == 12288 && train_set_x.cols() == 209);
	Assert(train_set_y.rows() == 1 && train_set_y.cols() == 209);
	Assert(test_set_x.rows() == 12288 && test_set_x.cols() == 50);
	Assert(test_set_y.rows() == 1 && test_set_y.cols() == 50);
	LRTrainingSet lrts;
	optimize(&lrs, &lrts, train_set_x, train_set_y, 1500, 0.005f);
	cout << "//=============================//" << endl;
	cout << "train accuracy: " << 100 - (Abs(predict(lrs, train_set_x) - train_set_y).sum() / train_set_x.cols() * 100) << endl;
	cout << "test accuracy: " << 100 - (Abs(predict(lrs, test_set_x) - test_set_y).sum() / test_set_x.cols() * 100) << endl;

	int imageIndex = 0;
	auto preds = predict(lrs, test_set_x);
	while(imageIndex<50) {
		if(preds(imageIndex) != test_set_y(imageIndex) ){
			for(int x = 0; x < 64; x++) {
				for(int y = 0; y < 192; y++) {
					float pix = test_set_x(x * 192 + y, imageIndex);
					cout << (pix < 0.1f ? ' ' : pix < 0.2f ? '.' : pix < 0.3f ? '-' : pix < 0.4f ? '+' : pix < 0.5f ? '*' : pix < 0.75f ? 'X' : '#');
				}
				cout << endl;
			}
			PrintBool(preds(imageIndex) == 1.f);
			PrintBool(test_set_y(imageIndex) == 1.f);
		}
		imageIndex++;
	}		
	cout << "Press a key to continue...";
	getchar();
}	