#pragma once

#include "windows.h"
#include "color.h"
#define Pi32 3.14159265359f


void drawLine(HDC *hdc, int x, int y, int x1, int y1, Color c) {
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
			SetPixelV(*hdc, x, y, RGB(c.r, c.g, c.b));
			y += dy;
			x += dx;
		} while(x != x1);
	} else if(dy) {
		if(dy) dx /= abs(dy);
		else dx = 0;
		if(dy > 0) dy = 1;
		else dy = -1;
		do { //for(;y<y1; y++){
			SetPixelV(*hdc, x, y, RGB(c.r, c.g, c.b));
			x += dx;
			y += dy;
		} while(y != y1);
	}
}

void DrawCircle(HDC *hdc, int x, int y, float d, Color c ) {
	double rad = d / 2.0;
	for(int n = 0; n < 32; n++) {
		double a = n*Pi32*4.0 / 32.0;
		double b = (n + 1)*Pi32*4.0 / 32.0;
		drawLine(hdc, int(x + sin(a)*rad), int(y + cos(a)*rad), int(x + sin(b)*rad), int(y + cos(b)*rad), c);
	}
}

void DrawFilledCircle(void *buffer, int bufferWidth, int x, int y, float d, Color c) {
	int r = int(d*0.5);
	for(int h = -r; h < r; h++) {
		int height = (int)sqrt(r * r - h * h);
		for(int v = -height; v < height; v++) {
			int *pixel = (int *)buffer + int(((x)+h) + ((y + v)* bufferWidth));
			*pixel = ((c.r << 16) | (c.g << 8) | c.b);
		}
	}
}