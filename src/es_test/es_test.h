#pragma once
#include "d_Matrix.h"
#include "d_math.h"
#include <Eigen/dense>
#include <iostream>

using namespace std;
using namespace Eigen;

static int verbosity = 2;
static const float thresholdMultiplier = (FLT_EPSILON * 0.5f);

d_Matrix to_device(MatrixXf matrix);
MatrixXf to_host(d_Matrix d_matrix);
void PrintHeader(string testType);
string testMultipy(int m, int n, int k);
string testTransposeRight(int m, int n, int k);
string testTransposeLeft(int m, int n, int k);
string testSum(int m, int k);
string testTranspose(int m, int k);
