#pragma once
#include "es_core_pch.h"

#define OUT_LINE(f, t, s) file << strTab(t) << s << std::endl;

template <class T>
void strCast(T *out, std::string str) {
	std::stringstream convertor(str);
	convertor >> *out;
}

inline bool strFind(std::string str, std::string token) {
	return str.find(token) != std::string::npos;
}

inline std::string strTab(int num) {
	std::string outStr;
	for (int i = 0; i < num; i++) {
		outStr += "\t";
	}
	return outStr;
}
