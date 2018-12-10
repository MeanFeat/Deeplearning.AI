#include "color.h"

int Color::ToBit() {
	return ((r << 16) | ((g << 8) | b));
}

Color Color::Blend(Color other, double percent) {
	return Color(unsigned char(double(r) + (percent*(double(other.r) - double(r)))),
				 unsigned char(double(g) + (percent*(double(other.g) - double(g)))),
				 unsigned char(double(b) + (percent*(double(other.b) - double(b)))),
				 unsigned char(double(a) + (percent*(double(other.a) - double(a)))));
}