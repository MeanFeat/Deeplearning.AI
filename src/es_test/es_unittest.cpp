#include "pch.h"
#include "CppUnitTest.h"
#include "es_test.h"
using namespace Microsoft::VisualStudio::CppUnitTestFramework;
#define NAME_RUN(n,arg) TEST_METHOD(n) { Assert::AreEqual(arg,true);}

namespace esunittest {
#ifndef UNITTEST_LISTS
#define UNITTEST_LISTS
	#include "test_unit.generated"
#endif
}

