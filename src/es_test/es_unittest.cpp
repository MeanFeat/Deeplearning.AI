#include "pch.h"
#include "CppUnitTest.h"
#include "es_test.cpp"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

#define NAME_RUN(n,arg) TEST_METHOD(n) {  ProcessMessage(arg); }

void ProcessMessage(testResult rst) {
	string err = rst.message;
	if (!rst.passed){
		err = "\n"+err;
		std::wstring widestr = std::wstring(err.begin(), err.end());
		Assert::Fail((widestr.c_str()));
	}
}

namespace esunittest {
#ifndef UNITTEST_LISTS
#define UNITTEST_LISTS
	#include GENERATED_UNIT_TESTS
#endif
}

