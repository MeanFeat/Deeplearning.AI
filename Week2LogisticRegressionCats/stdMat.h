#pragma once
#include <Eigen/dense>
#include <vector>

using namespace Eigen;
using namespace std;

inline float UnarySig(float val){ return 1 / (1 + std::exp(-val)); }
inline MatrixXf Sigmoid(MatrixXf mat){ return mat.unaryExpr(&UnarySig); }

inline float UnaryTanh(float val){ return std::tanh(val); }
inline MatrixXf Tanh(MatrixXf mat){ return mat.unaryExpr(&UnaryTanh); }

inline float UnaryLog(float val) { return std::log(val); }
inline MatrixXf Log(MatrixXf mat) { return mat.unaryExpr(&UnaryLog); }

inline float UnaryRound(float val) { return val <= 0.5f ? 0.f : 1.f; }
inline MatrixXf Round(MatrixXf mat) { return mat.unaryExpr(&UnaryRound); }

inline float UnaryAbs(float val) { return abs(val); }
inline MatrixXf Abs(MatrixXf mat) { return mat.unaryExpr(&UnaryAbs); }