
#include "stdafx.h"
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <SDL.h>

#define WINWIDTH 800
#define WINHEIGHT 600

struct LogRegSet {
	MatrixXf w;
	float b;
};

struct LRTrainingSet {
	MatrixXf dw;
	float db;
	float Cost;
};

LogRegSet InitToZero(int dim) {
	LogRegSet Result;
	Result.w = MatrixXf::Zero(dim, 1);
	Result.b = 0.f;
	return Result;
}

MatrixXf GetModelOutput(LogRegSet set, MatrixXf X) {
	MatrixXf biases = MatrixXf::Ones(1, X.cols()) * set.b;
	return Sigmoid(set.w.transpose() * X + biases);
}

LRTrainingSet propegate(LogRegSet set, MatrixXf X, MatrixXf Y) {
	LRTrainingSet Result;
	int m = (int)Y.cols();
	float coeff = 1.f / m;
	MatrixXf A = GetModelOutput(set, X);
	Result.Cost = -((Y.cwiseProduct(Log(A))) + (MatrixXf::Ones(1, m) - Y).cwiseProduct((Log(MatrixXf::Ones(1, m) - A)))).sum() * coeff;
	Result.dw = (X * (A - Y).transpose()) * coeff;
	Result.db = (A - Y).sum()*coeff;
	return Result;
}

MatrixXf predict(LogRegSet lrs, MatrixXf X) {
	return (GetModelOutput(lrs, X)).array().round();
}

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

void optimize(SDL_Renderer *ren, LogRegSet *lrsPtr, LRTrainingSet *trainSetPtr, MatrixXf X, MatrixXf Y, int iterations, float learnRate) {
	vector<float> Xplots;
	vector<float> tXplots;
	SDL_SetRenderDrawColor(ren, 0, 0, 0, 0);
	SDL_RenderClear(ren);
	for(int i = 0; i < iterations; i++) {
		*trainSetPtr = propegate(*lrsPtr, X, Y);
		lrsPtr->w = lrsPtr->w - (learnRate * trainSetPtr->dw);
		lrsPtr->b = lrsPtr->b - (learnRate * trainSetPtr->db);
		Assert(trainSetPtr->dw.rows() == lrsPtr->w.rows() && trainSetPtr->dw.cols() == lrsPtr->w.cols());
		if(i % 2 == 0) {
			SDL_SetRenderDrawColor(ren, 0, 0, 0, 0);
			SDL_RenderClear(ren);
			Xplots.push_back(((predict(*lrsPtr, X) - Y).array().abs().sum() / Y.cols() * WINHEIGHT));
			int slider = Xplots.size() >= WINWIDTH ? Xplots.size() - WINWIDTH : 0;
			for(int p = slider + 1; p < (int)Xplots.size(); p++) {
				SDL_SetRenderDrawColor(ren, 0, 0, 255, 255);
				drawLine(ren, (p - 1) - slider, int(Xplots[p - 1]), p - slider, int(Xplots[p]));
			}
			SDL_RenderPresent(ren);
		}
	}
}

void LogisticRegression(SDL_Renderer *ren) {
	LogRegSet lrs = InitToZero(12288);
	MatrixXf train_set_x;
	read_binary("catTrain.dat", train_set_x);
	MatrixXf test_set_x;
	read_binary("catTest.dat", test_set_x);
	MatrixXf train_set_y;
	read_binary("catTrainLabels.dat", train_set_y);
	MatrixXf test_set_y;
	read_binary("catTestLabels.dat", test_set_y);
	Assert(train_set_x.rows() == 12288 && train_set_x.cols() == 209);
	Assert(train_set_y.rows() == 1 && train_set_y.cols() == 209);
	Assert(test_set_x.rows() == 12288 && test_set_x.cols() == 50);
	Assert(test_set_y.rows() == 1 && test_set_y.cols() == 50);
	LRTrainingSet lrts;
	optimize(ren, &lrs, &lrts, train_set_x, train_set_y, 1500, 0.005f);

	SDL_SetRenderDrawColor(ren, 0, 0, 0, 0);
	SDL_RenderClear(ren);
	int imageIndex = 0;
	int offset = 0;
	int offsetIndex = 0;
	auto preds = predict(lrs, test_set_x);
	while(imageIndex < test_set_y.cols()) {
		if(preds(imageIndex) != test_set_y(imageIndex)) {
			for(int x = 0; x < 64; x++) {
				for(int y = 0; y < 192; y += 3) {
					SDL_SetRenderDrawColor(ren, test_set_x(x * 192 + y, imageIndex) * 255,
										   test_set_x(x * 192 + y + 1, imageIndex) * 255,
										   test_set_x(x * 192 + y + 2, imageIndex) * 255,
										   255);
					SDL_RenderDrawPoint(ren, offset + y / 3, offsetIndex * 64 + x);
				}
			}
			offsetIndex++;
		}
		imageIndex++;
		if(offsetIndex >= 8) {
			offset += 70;
			offsetIndex = 0;
		}
	}
	SDL_RenderPresent(ren);
}

int main(int argc, char *argv[]) {
	SDL_Window *win;
	SDL_Renderer *ren;
	SDL_Init(SDL_INIT_VIDEO);
	SDL_CreateWindowAndRenderer(WINWIDTH, WINHEIGHT, 0, &win, &ren);
	LogisticRegression(ren);
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