#pragma once
#include "stdMat.h"
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

void DrawCircle(HDC *hdc, int x, int y, float d, Color c) {
	double rad = d / 2.0;
	for(int n = 0; n < 32; n++) {
		double a = n*Pi32*4.0 / 32.0;
		double b = (n + 1)*Pi32*4.0 / 32.0;
		drawLine(hdc, int(x + sin(a)*rad), int(y + cos(a)*rad), int(x + sin(b)*rad), int(y + cos(b)*rad), c);
	}
}

void DrawLine(void *buffer, int bufferWidth, float aX, float aY, float bX, float bY, Color col) {
	float dx = bX - aX;
	float dy = bY - aY;
	if(abs(dx) > abs(dy) && dx) {
		if(dx) dy /= abs(dx);
		else dy = 0.f;
		if(dx >= 0.f) dx = 1.f;
		else dx = -1.f;
		do { //for(;x<x1; x++){
			int *pixel = (int *)buffer + int(aX + int(aY) * bufferWidth);
			*pixel = ((col.r << 16) | (col.g << 8) | col.b);
			aY += dy;
			aX += dx;
		} while(aX != bX);
	} else if(dy) {
		if(dy) dx /= abs(dy);
		else dx = 0;
		if(dy > 0) dy = 1.f;
		else dy = -1.f;
		do { //for(;y<y1; y++){
			int *pixel = (int *)buffer + int(aX + aY * bufferWidth);
			*pixel = ((col.r << 16) | (col.g << 8) | col.b);
			aY += dy;
			aX += dx;
		} while(aY != bY);
	}
}

void DrawHistory(void *buffer, int bufferWidth, vector<float> hist) {
	float compressor = int(hist.size() - 1) > bufferWidth ? float(bufferWidth) / float(hist.size() - 1) : 1.f;
	for(int sample = 1; sample < (int)hist.size() - 1; sample++) {
		DrawLine(buffer, bufferWidth, (sample - 1) * compressor,
				hist[sample - 1], sample * compressor, 
				hist[sample], Color(200, 90, 90, 255));
	}
}

void DrawFilledCircle(void *buffer, int bufferWidth, int x, int y, float d, Color c) {
	int r = int(d*0.5);
	for(int h = -r; h < r; h++) {
		int height = (int)sqrt(r * r - h * h);

		if(x - d > 0 && x + d < 800 && y - d > 0 && y + d < bufferWidth) {
			for(int v = -height; v < height; v++) {
				int *pixel = (int *)buffer + int(((x)+h) + ((y + v)* bufferWidth));
				*pixel = ((c.r << 16) | (c.g << 8) | c.b);
			}
		}
	}
}