#pragma once
#include "es_test.h"
#include "es_parser.h"
#include <conio.h>
#include <fstream>
#include <iostream>

enum class TestParseState {
	functionName,
	header,
	prefix,
	args,
	none
};