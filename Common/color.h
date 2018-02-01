#pragma once

class Color {
public:
	Color(int inR, int inG, int inB, int inA) : r(inR), g(inG), b(inB), a(inA) {}
	int r, g, b, a;
	int ToBit();
	Color Blend(Color other, float percent);

};

int Color::ToBit() {
	return ((r << 16) | ((g << 8) | b));
}
Color Color::Blend(Color other, float percent) {
	return Color(unsigned char(float(r) + (percent*(float(other.r) - float(r)))),
				 unsigned char(float(g) + (percent*(float(other.g) - float(g)))),
				 unsigned char(float(b) + (percent*(float(other.b) - float(b)))),
				 unsigned char(float(a) + (percent*(float(other.a) - float(a)))));
}