#pragma once

#include "SDL.h"
#include "color.h"


void drawLine(SDL_Renderer *ren, int x, int y, int x1, int y1) {
	int dx = x1 - x;
	int dy = y1 - y;
	if(dx == 0 || dy == 0) {
		//return;
	}
	if(abs(dx) > abs(dy) && dx) {
		if(dx) dy /= abs(dx);
		else dy = 0;
		if(dx >= 0) dx = 1;
		else dx = -1;
		do { //for(;x<x1; x++){
			SDL_RenderDrawPoint(ren, x, y);
			y += dy;
			x += dx;
		} while(x != x1);
	} else if(dy) {
		if(dy) dx /= abs(dy);
		else dx = 0;
		if(dy > 0) dy = 1;
		else dy = -1;
		do { //for(;y<y1; y++){
			SDL_RenderDrawPoint(ren, x, y);
			x += dx;
			y += dy;
		} while(y != y1);
	}
}

void DrawCircle(SDL_Renderer *ren, int x, int y, float d, Color c ) {
	SDL_SetRenderDrawColor(ren,c);
	double rad = d / 2.0;
	for(int n = 0; n < 32; n++) {
		double a = n*M_PI*4.0 / 32.0;
		double b = (n + 1)*M_PI*4.0 / 32.0;
		drawLine(ren, int(x + sin(a)*rad), int(y + cos(a)*rad), int(x + sin(b)*rad), int(y + cos(b)*rad));
	}
}

void DrawFilledCircle(SDL_Renderer *ren, int x, int y, float d, Color c) {
	SDL_SetRenderDrawColor(ren, c);
	int r = int(d*0.5);
	for(int h = -r; h < r; h++) {
		int height = (int)sqrt(r * r - h * h);
		for(int v = -height; v < height; v++)
			SDL_RenderDrawPoint(ren, x+h, y+v);
	}
}
