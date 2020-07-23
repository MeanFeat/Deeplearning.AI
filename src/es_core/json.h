#pragma once

enum ParseState{
	 none = 0,
	 layer,
	 activation,
	 weightShape,
	 biasShape,
	 weightValues,
	 biasValues,
};

template <class T>
void strCast(T *out, std::string str) {
	std::stringstream convertor(str);
	convertor >> *out;
}

bool strFind(std::string str, std::string token) {
	return str.find(token) != std::string::npos;
}

Activation ReadActivation(std::string str) {
	if( str.find("tanh") != std::string::npos ) {
		return Activation::Tanh;
	}
	return Activation::Linear;
}