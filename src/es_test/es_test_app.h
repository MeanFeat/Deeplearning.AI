#pragma once
#include "es_test.h"
#include <conio.h>
#include <fstream>
#include <iostream>
using namespace std;

#define GENERATED_TESTS "test_cpp.generated"
#define GENERATED_TEMP_TESTS "test_temp_cpp.generated"
enum ParseState {
	functionName,
	header,
	prefix,
	args,
	none
};

template <class T>
void strCast(T *out, std::string str) {
	std::stringstream convertor(str);
	convertor >> *out;
}

bool strFind(std::string str, std::string token) {
	return str.find(token) != std::string::npos;
}

