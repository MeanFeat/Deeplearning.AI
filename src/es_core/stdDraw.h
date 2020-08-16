#pragma once
#include "es_core_pch.h"
#include "stdMatrix.h"
#include "color.h"

#define Pi32 3.14159265359f

static Color positiveColor = Color(100, 167, 211, 255);
static Color negativeColor = Color(255, 184, 113, 255);

struct Buffer {
	void* memory;
	int width;
	int height;
	int titleOffset;
	BITMAPINFO bitmapInfo = { 0 };
};

void drawLine(HDC *hdc, int x, int y, int x1, int y1, Color c) {
	int dx = x1 - x;
	int dy = y1 - y;
	if (dx == 0 || dy == 0) {
		//return;
	}
	if (abs(dx) > abs(dy) && dx) {
		if (dx) dy /= abs(dx);
		else dy = 0;
		if (dx >= 0) dx = 1;
		else dx = -1;
		do { //for(;x<x1; x++){
			SetPixelV(*hdc, x, y, RGB(c.r, c.g, c.b));
			y += dy;
			x += dx;
		} while (x != x1);
	}
	else if (dy) {
		if (dy) dx /= abs(dy);
		else dx = 0;
		if (dy > 0) dy = 1;
		else dy = -1;
		do { //for(;y<y1; y++){
			SetPixelV(*hdc, x, y, RGB(c.r, c.g, c.b));
			x += dx;
			y += dy;
		} while (y != y1);
	}
}

void DrawCircle(HDC *hdc, int x, int y, float d, Color c) {
	float rad = d / 2.f;
	for (int n = 0; n < 32; n++) {
		float a = n * Pi32*4.f / 32.f;
		float b = (n + 1)*Pi32*4.f / 32.f;
		drawLine(hdc, int(x + sin(a)*rad), int(y + cos(a)*rad), int(x + sin(b)*rad), int(y + cos(b)*rad), c);
	}
}

void DrawLine(Buffer buffer, float aX, float aY, float bX, float bY, Color col) {
	float dx = bX - aX;
	float dy = bY - aY;
	if (abs(dx) > abs(dy) && dx) {
		if (dx) dy /= abs(dx);
		else dy = 0.f;
		if (dx >= 0.f) dx = 1;
		else dx = -1;
		do { //for(;x<x1; x++){
			int *pixel = (int *)buffer.memory + int(aX + int(aY) * buffer.width);
			*pixel = ((col.r << 16) | (col.g << 8) | col.b);
			aY += dy;
			aX += dx;
		} while (aX != bX);
	}
	else if (dy) {
		if (dy) dx /= abs(dy);
		else dx = 0;
		if (dy > 0) dy = 1;
		else dy = -1;
		do { //for(;y<y1; y++){
			int *pixel = (int *)buffer.memory + int(aX + int(aY) * buffer.width);
			*pixel = ((col.r << 16) | (col.g << 8) | col.b);
			aY += dy;
			aX += dx;
		} while (aY != bY);
	}
}

void DrawHistory(Buffer buffer, std::vector<float> hist, Color c) {
	float compressor = int(hist.size()) > buffer.width ? float(buffer.width) / float(hist.size()) : 1.f;
	for (int sample = 1; sample < (int)hist.size() - 1; sample++) {
		DrawLine(buffer, float(int((sample - 1) * compressor) - buffer.width),
			floor(hist[sample - 1]), float(int(sample * compressor) - buffer.width),
			floor(hist[sample]), c);
	}
}

void DrawFilledCircle(Buffer buffer, int x, int y, float d, Color c) {
	int r = int(d*0.5f);
	for (int h = -r; h < r; h++) {
		int height = (int)sqrt(r * r - h * h);

		if (x - d > 0 && x + d < 800 && y - d > 0 && y + d < buffer.width) {
			for (int v = -height; v < height; v++) {
				int *pixel = (int *)buffer.memory + int(((x)+h) + ((y + v)* buffer.width));
				*pixel = ((c.r << 16) | (c.g << 8) | c.b);
			}
		}
	}
}

Eigen::MatrixXf BuildDisplayCoords(Buffer buffer, float scale = 1.f) {
	Eigen::MatrixXf out(buffer.width * buffer.height, 2);
	Eigen::VectorXf row(buffer.width);
	Eigen::VectorXf cols(buffer.width * buffer.height);
	int halfWidth = int(buffer.width * 0.5f);
	for (int x = 0; x < buffer.width; ++x) {
		row(x) = float((x - halfWidth) / scale);
	}
	for (int y = 0; y < buffer.height; ++y) {
		for (int x = 0; x < buffer.width; ++x) {
			cols(y*buffer.width + x) = float((y - halfWidth) / scale);
		}
	}
	out << row.replicate(buffer.height, 1), cols;
	out.col(1) *= -1.f;
	return out;
}

void FillScreen(Buffer buff, Color col = Color(0, 0, 0, 0)) {
	int *pixel = (int *)buff.memory;
	for (int i = 0; i < buff.width * buff.height; i += 4) {
		*pixel++ = col.ToBit();
		*pixel++ = col.ToBit();
		*pixel++ = col.ToBit();
		*pixel++ = col.ToBit();
	}
}

void ClearScreen(Buffer buff) {
	FillScreen(buff);
}