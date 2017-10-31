
#include "stdMat.h"
#include "stdDraw.h"
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <SDL.h>

#define WINWIDTH 800
#define WINHEIGHT 800

void Plot(SDL_Renderer *ren, MatrixXf X, MatrixXf Y) {
	for (int i = 0; i < X.cols(); i++) {
		Assert(X(1, i) != X(0, i));
		SDL_SetRenderDrawColor(ren, 255, 255, 255, 255);
		DrawFilledCircle(ren, (WINWIDTH / 2) + X(0, i) * 80, (WINHEIGHT / 2) + X(1, i) * 80, 8.f, Color(0, 0, 0, 255));
		DrawFilledCircle(ren, (WINWIDTH / 2) + X(0, i) * 80, (WINHEIGHT / 2) + X(1, i) * 80, 6.f, 
						 Y(i) == 1 ? Color(0, 255, 255, 255) : Color(255, 0, 0, 255));
	}
	SDL_RenderPresent(ren);
}

int main(int argc, char *argv[]) {
	MatrixXf X;// = (MatrixXf)BuildMatFromFile("new.txt"); write_binary("planar.dat", X);
	MatrixXf Y;// = (MatrixXf)BuildMatFromFile("newL.txt"); write_binary("planarLabels.dat", Y);
	read_binary("planar.dat", X);
	read_binary("planarLabels.dat", Y);

	cout << Y << endl;
	
	SDL_Window *win;
	SDL_Renderer *ren;
	SDL_Init(SDL_INIT_VIDEO);
	SDL_CreateWindowAndRenderer(WINWIDTH, WINHEIGHT, 0, &win, &ren);
	SDL_SetRenderDrawColor(ren, 255, 255, 255, 255);
	SDL_RenderClear(ren);

	Plot(ren, X, Y);

	SDL_Event localEvent;
	SDL_PollEvent(&localEvent);
	while(localEvent.type != SDL_MOUSEBUTTONDOWN) {
		SDL_PollEvent(&localEvent);
	}
	SDL_DestroyRenderer(ren);
	SDL_DestroyWindow(win);
	SDL_Quit();
	return 0;
}