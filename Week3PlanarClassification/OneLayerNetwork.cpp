
#include "stdMat.h"
#include "stdDraw.h"
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <SDL.h>

#define WINWIDTH 400
#define WINHEIGHT 400
#define WINHALFWIDTH WINWIDTH * 0.5f
#define WINHALFHEIGHT WINHEIGHT * 0.5f
#define SCALE 40
#define SAMPLEDIM = 6

void PlotData(SDL_Renderer *ren, MatrixXf X, MatrixXf Y) {
	for(int i = 0; i < X.cols(); i++) {
		Assert(X(1, i) != X(0, i));
		DrawFilledCircle(ren, int((WINWIDTH / 2) + X(0, i) * SCALE), int((WINHEIGHT / 2) + -X(1, i) * SCALE), SCALE * 0.275f, Color(0, 0, 0, 255));
		DrawFilledCircle(ren, int((WINWIDTH / 2) + X(0, i) * SCALE), int((WINHEIGHT / 2) + -X(1, i) * SCALE), SCALE * 0.2f,
						 Y(i) == 1 ? Color(0, 255, 255, 255) : Color(255, 0, 0, 255));
	}
	SDL_RenderPresent(ren);
}

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

int main(int argc, char *argv[]) {
	MatrixXf X;// = (MatrixXf)BuildMatFromFile("new.txt"); write_binary("planar.dat", X);
	MatrixXf Y;// = (MatrixXf)BuildMatFromFile("newL.txt"); write_binary("planarLabels.dat", Y);
	read_binary("planar.dat", X);
	read_binary("planarLabels.dat", Y);
	SDL_Window *win;
	SDL_Renderer *ren;
	SDL_Init(SDL_INIT_VIDEO);
	SDL_CreateWindowAndRenderer(WINWIDTH, WINHEIGHT, 0, &win, &ren);
	SDL_Event localEvent;
	MatrixXf temp = BuildDisplayCoords();
	bool running = true;
	int frame = 0;
	while(running) {
		SDL_PollEvent(&localEvent);
		SDL_SetRenderDrawColor(ren, frame++, rand() % 255, 150, 255);
		for(int i = 0; i < temp.rows(); i++) {
			SDL_RenderDrawPoint(ren, int(temp(i, 0) + WINHALFHEIGHT), int((temp(i, 1))+ WINHALFWIDTH));
		} 
		PlotData(ren, X, Y);
		if(localEvent.type == SDL_WINDOWEVENT) {
			switch(localEvent.window.event) {
			case SDL_WINDOWEVENT_CLOSE:
				running = false;
				SDL_Log("Window %d closed", localEvent.window.windowID);
				SDL_DestroyRenderer(ren);
				SDL_DestroyWindow(win);
				SDL_Quit();
				break;
			}
		}
	}
	return EXIT_SUCCESS;
}