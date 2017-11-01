#include "stdafx.h"
#include "CppUnitTest.h"
#include "..\Common\stdMat.h"
#include <Eigen/dense>
#include "../Week2LogisticRegressionCats/LogReg.cpp"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

#define Tollerance 0.0000001

namespace Week2UnitTest {
	TEST_CLASS(UnitTest1) {
public:
	TEST_METHOD(SigmoidTest) {
		MatrixXd t(2, 1); t << 0.0, 2.0;
		MatrixXd e(2, 1); e << 0.5, 0.88079708;
		if(!Sigmoid(t).isApprox(e, Tollerance)) {
			Assert::Fail(0);
		}
	}
	TEST_METHOD(PropegateTest) {
		MatrixXd w(2, 1); w << 1.0, 2.0;
		double b = 2.0;
		MatrixXd X(2, 3); X.row(0) << 1.0, 2.0, -1.0; X.row(1) << 3.0, 4.0, -3.2;
		MatrixXd Y(1, 3); Y << 1.0, 0.0, 1.0;
		MatrixXd edw(2, 1); edw << 0.99845601, 2.39507239;
		double edb = 0.00145557813678;
		double eCost = 5.80154531939;
		LogRegSet lrs;
		lrs.w = w; lrs.b = b;
		LRTrainingSet lrts;
		lrts = propegate(lrs, X, Y);
		if(!lrts.dw.isApprox(edw, Tollerance) || abs(lrts.db - edb) > Tollerance || abs(lrts.Cost - eCost) > Tollerance) {
			Assert::Fail(0);
		}
	}
	};
}
