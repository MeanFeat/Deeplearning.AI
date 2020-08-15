#pragma once
#include "es_test.h"
#include "es_parser.h"
#include <conio.h>
#include <fstream>
#include <iostream>
using namespace std;

enum class TestParseState {
	functionName,
	header,
	prefix,
	args,
	none
};