#pragma once
#include "es_test.h"
#include "es_parser.h"

enum class TestParseState {
	category,
	functionName,
	header,
	prefix,
	args,
	none
};