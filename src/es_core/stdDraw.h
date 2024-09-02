#pragma once
#include "es_core_pch.h"
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

void drawLine(HDC *hdc, int x, int y, int x1, int y1, Color c);

void DrawCircle(HDC *hdc, int x, int y, float d, Color c);

void DrawLine(Buffer buffer, float aX, float aY, float bX, float bY, Color col);

void DrawHistory(Buffer buffer, std::vector<float> hist, Color c);

void DrawFilledCircle(Buffer buffer, int x, int y, float d, Color c);

Eigen::MatrixXf BuildDisplayCoords(Buffer buffer, float scale = 1.f);

void FillScreen(Buffer buff, Color col = Color(0, 0, 0, 0));

void ClearScreen(Buffer buff);
