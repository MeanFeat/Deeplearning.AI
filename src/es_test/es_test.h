#pragma once
#include "stdNet.h"
#include "d_Matrix.h"
#include "d_math.h"
#include <Eigen/dense>
#include <iostream>

#define GENERATED_TESTS "tests_cpp.generated"
#define GENERATED_UNIT_TESTS "tests_unit.generated"

using namespace std;
using namespace Eigen;

static int verbosity = 2;
static const float thresholdMultiplier = (FLT_EPSILON) * 2.f;

d_Matrix to_device(MatrixXf matrix);
MatrixXf to_host(d_Matrix d_matrix);
void PrintHeader(string testType);
string testMultipy(int m, int n, int k);
string testTransposeRight(int m, int n, int k);
string testTransposeLeft(int m, int n, int k);
string testSum(int m, int k);
string testTranspose(int m, int k);
string testMultScalar(int m, int k);
string testAdd(int m, int k);
string testSubtract(int m, int k);
string testSquare(int m, int k);
string testSigmoid(int m, int k);
string testTanh(int m, int k);
string testReLU(int m, int k);
string testLReLU(int m, int k);