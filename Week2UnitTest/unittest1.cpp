#include "stdafx.h"
#include "CppUnitTest.h"
#include "..\Common\stdMat.h"
#include <Eigen/dense>
#include "../Week2LogisticRegressionCats/LogReg.cpp"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

#define Tollerance 0.000001f

namespace Week2UnitTest {
	TEST_CLASS(UnitTest1) {
public:
	TEST_METHOD(SigmoidTest) {
		MatrixXf t(2, 1); t << 0.0f, 2.0f;
		MatrixXf e(2, 1); e << 0.5f, 0.88079708f;
		if(!Sigmoid(t).isApprox(e, Tollerance)) {
			Assert::Fail(0);
		}
	}
	TEST_METHOD(PropegateTest) {
		MatrixXf w(2, 1); w << 1.0f, 2.0f;
		float b = 2.0f;
		MatrixXf X(2, 3); X.row(0) << 1.0f, 2.0f, -1.0; X.row(1) << 3.0f, 4.0f, -3.2f;
		MatrixXf Y(1, 3); Y << 1.0f, 0.0f, 1.0f;
		MatrixXf edw(2, 1); edw << 0.99845601f, 2.39507239f;
		float edb = 0.00145557813678f;
		double eCost = 5.79859066; // Different value from course because of float / double precision when calulating cost
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
