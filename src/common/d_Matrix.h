#pragma once
#include "d_cudahelpers.h"
class d_Matrix {
public:
	d_Matrix();
	d_Matrix(int rows, int cols);
	d_Matrix(const float *host_data, int rows, int cols);
	d_Matrix(const d_Matrix& other);
	d_Matrix& operator=(const d_Matrix& other);
	~d_Matrix();
	float* &d_data() { return device_data; }
	float* d_data() const { return device_data; }
	int rows() { return rowCount; }
	int cols() { return colCount; }
	int rows() const { return rowCount; }
	int cols() const { return colCount; }
	int size() { return rowCount * colCount; }
	int size() const { return rowCount * colCount; }
	d_Matrix serialize();
	d_Matrix serialize() const;
	void serializeInPlace();
	void setShape(int rows, int cols);
	size_t memSize() { return size() * sizeof(float); }
	size_t memSize() const { return size() * sizeof(float); }
	void free() const;
private:
	int rowCount;
	int colCount;
	float* device_data;
};
