#include "pch.h"
#include "es_test.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

#define NAME_RUN(n,arg) TEST_METHOD(n) {  ProcessMessage(arg); }

void ProcessMessage(testResult rst) {
	std::string err = rst.message;
	if (!rst.passed) {
		err = "\n" + err;
		std::wstring widestr = std::wstring(err.begin(), err.end());
		Assert::Fail((widestr.c_str()));
	}
}

#ifndef UNITTEST_LISTS
#define UNITTEST_LISTS
#include GENERATED_UNIT_TESTS
#endif