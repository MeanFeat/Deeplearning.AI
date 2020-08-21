#pragma once
#include "es_core_pch.h"
#include <regex>

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

inline std::string strClean(std::string s){
	s.erase(remove(s.begin(), s.end(), ' '), s.end());
	s.erase(remove(s.begin(), s.end(), '\t'), s.end());
	return s;
}

inline std::string strReplace(std::string s, std::string f, std::string r){
	std::string out = s;
	size_t index = 0;
	int sl = (int)strlen(f.c_str());
	while (true) {
		index = out.find(f, index);
		if (index == std::string::npos) break;
		out.replace(index, sl, r);
		index += sl;
	}
	return out;
}
