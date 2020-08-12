#include "pch.h"
#include "CppUnitTest.h"
#include "es_test.h"
string err;
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

#define NAME_RUN(n,arg) TEST_METHOD(n) {  ProcessMessage(arg); }

void ProcessMessage(string err) {
	if ((err.find("Fail:") != std::string::npos)){
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

