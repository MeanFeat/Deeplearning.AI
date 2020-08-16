#pragma once
#include "es_core_pch.h"
template <class T>
void strCast(T *out, std::string str) {
	std::stringstream convertor(str);
	convertor >> *out;
}

inline bool strFind(std::string str, std::string token) {
	return str.find(token) != std::string::npos;
}