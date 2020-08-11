#pragma once
#include <iostream>
#include <Eigen/dense>
#include "d_Matrix.h"
#include "d_math.h"
using namespace std;
using namespace Eigen;

static int verbosity = 1;
static const float thresholdMultiplier = (FLT_EPSILON * 0.5f);

d_Matrix to_device(MatrixXf matrix);
MatrixXf to_host(d_Matrix d_matrix);
void PrintHeader(string testType);
bool testMultipy(int m, int n, int k);
bool testTransposeRight(int m, int n, int k);
bool testTransposeLeft(int m, int n, int k);
bool testSum(int m, int k);
bool testTranspose(int m, int k);
