
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <Eigen/dense>
#include "win32_DeepLearning.h"

#define WINWIDTH 300
#define WINHEIGHT 300
#define WINHALFWIDTH WINWIDTH * 0.5f
#define WINHALFHEIGHT WINHEIGHT * 0.5f
#define SCALE 30
using namespace Eigen;
//void PlotData(SDL_Renderer *ren, MatrixXf X, MatrixXf Y) {
//	for(int i = 0; i < X.cols(); i++) {
//		Assert(X(1, i) != X(0, i));
//		DrawFilledCircle(ren, int((WINWIDTH / 2) + X(0, i) * SCALE), int((WINHEIGHT / 2) + -X(1, i) * SCALE), SCALE * 0.275f, Color(0, 0, 0, 255));
//		DrawFilledCircle(ren, int((WINWIDTH / 2) + X(0, i) * SCALE), int((WINHEIGHT / 2) + -X(1, i) * SCALE), SCALE * 0.2f,
//						 Y(i) == 1 ? Color(0, 255, 255, 255) : Color(255, 0, 0, 255));
//	}
//	SDL_RenderPresent(ren);
//}

MatrixXf BuildDisplayCoords() {
	MatrixXf out(WINWIDTH * WINHEIGHT, 2);
	VectorXf row(WINWIDTH);
	VectorXf cols(WINWIDTH * WINHEIGHT);
	for(int x = 0; x < WINWIDTH; x++) {
		row(x) = float(x - WINHALFWIDTH);
	}
	for(int y = 0; y < WINHEIGHT; y++) {
		for(int x = 0; x < WINWIDTH; x++){
			cols(y*WINWIDTH+x) = float((y - WINHALFWIDTH));;
		}
	}
	out << row.replicate(WINHEIGHT, 1), cols;
	return out;
}


