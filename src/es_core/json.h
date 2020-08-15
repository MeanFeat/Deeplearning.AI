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

Activation ReadActivation(string str) {
	if( strFind(str, "Sigmoid") ) {
		return Activation::Sigmoid;
	} else if( strFind(str, "Tanh") ) {
		return Activation::Tanh;
	} else if( strFind(str, "ReLU") ) {
		return Activation::ReLU;
	} else if( strFind(str, "LReLU") ) {
		return Activation::LReLU;
	} else {
		return Activation::Linear;
	}
}