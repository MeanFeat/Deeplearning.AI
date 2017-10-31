
#include "stdMat.h"
#include "stdDraw.h"
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <SDL.h>

#define WINWIDTH 800
#define WINHEIGHT 800
#define SCALE 80

void Plot(SDL_Renderer *ren, MatrixXf X, MatrixXf Y) {
	for (int i = 0; i < X.cols(); i++) {
		Assert(X(1, i) != X(0, i));
		SDL_SetRenderDrawColor(ren, 255, 255, 255, 255);
		DrawFilledCircle(ren, int((WINWIDTH / 2) + X(0, i) * SCALE), int((WINHEIGHT / 2) + -X(1, i) * SCALE), 14.f, Color(0, 0, 0, 255));
		DrawFilledCircle(ren, int((WINWIDTH / 2) + X(0, i) * SCALE), int((WINHEIGHT / 2) + -X(1, i) * SCALE), 13.f, 
						 Y(i) == 1 ? Color(0, 255, 255, 255) : Color(255, 0, 0, 255));
	}
	SDL_RenderPresent(ren);
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
	SDL_PollEvent(&localEvent);
	bool running = true;
	while(running) {
		SDL_SetRenderDrawColor(ren, 255, 255, 255, 255);
		SDL_RenderClear(ren);
		Plot(ren, X, Y);
		SDL_PollEvent(&localEvent);
		if(localEvent.type == SDL_WINDOWEVENT) {
			switch(localEvent.window.event) {
			case SDL_WINDOWEVENT_CLOSE:
				running = false;
				SDL_Log("Window %d closed", localEvent.window.windowID);
				break;
			}
		}
	}
	SDL_DestroyRenderer(ren);
	SDL_DestroyWindow(win);
	SDL_Quit();
	return 0;
}