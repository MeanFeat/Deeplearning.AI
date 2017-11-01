#pragma once
#include <Eigen/dense>
#include <vector>
#include <iostream>
#include <fstream>

#define Assert(Expression) if(!(Expression)) {*(int *)0 = 0;}
using namespace Eigen;
using namespace std;

inline MatrixXf Sigmoid(MatrixXf in) {
	return ((-1.0f*in).array().exp() + 1 ).cwiseInverse(); //faster than unary
}

inline MatrixXf Log(MatrixXf in) { 
	return in.array().log();
}

typedef Matrix<float, Dynamic, Dynamic> MatrixDynamic;
namespace Eigen {

	template<class Matrix>
	void write_binary(const char* filename, const Matrix& matrix) {
		std::ofstream out(filename, ios::out | ios::binary | ios::trunc);
		typename Matrix::Index rows = matrix.rows(), cols = matrix.cols();
		out.write((char*)(&rows), sizeof(typename Matrix::Index));
		out.write((char*)(&cols), sizeof(typename Matrix::Index));
		out.write((char*)matrix.data(), rows*cols * sizeof(typename Matrix::Scalar));
		out.close();
	}
	template<class Matrix>
	void read_binary(const char* filename, Matrix& matrix) {
		std::ifstream in(filename, ios::in | std::ios::binary);
		typename Matrix::Index rows = 0, cols = 0;
		in.read((char*)(&rows), sizeof(typename Matrix::Index));
		in.read((char*)(&cols), sizeof(typename Matrix::Index));
		matrix.resize(rows, cols);
		in.read((char *)matrix.data(), rows*cols * sizeof(typename Matrix::Scalar));
		in.close();
	}

	MatrixXf BuildMatFromFile(string fName) {
		vector<vector<float>> tempMat;
		ifstream file(fName);
		std::string line;
		int count = 0;
		float temp;
		int row = 0;
		while (file.good()) {
			vector<float> row;
			std::getline(file, line);
			std::stringstream iss(line); 
			std::string val;
			while (iss.good()) {
				std::getline(iss, val, ','); 
				std::stringstream convertor(val);
				convertor >> temp;
				row.push_back(temp);
			}
			tempMat.push_back(row);
		}
		MatrixXf outMat(tempMat.size(), tempMat[0].size()); 
		for (int i =0; i < (int)tempMat.size(); i++){
			for(int j = 0; j < (int)tempMat[i].size(); j++) {
				outMat(i, j) = tempMat[i][j];
			}
		}
		return outMat;
	}
}