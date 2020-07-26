#include "win32_ExpertSystems.h"
#include "stdMat.h"
#include "windowsx.h"
#include <time.h>

#define WINWIDTH 800
#define WINHEIGHT WINWIDTH
#define WINHALFWIDTH int((WINWIDTH-1)*0.5f)
#define WINHALFHEIGHT WINHALFWIDTH
#define MAXCAPTURECOUNT 25
#define MOUSETRAILLENGTH 50
#define CAPTURETHRESHOLD 50.f

global_variable bool globalRunning = true;
global_variable bool isVerifying = false;
global_variable bool skipStep = false;

global_variable vector<Vector2f> mouseCapture;
global_variable vector<Vector2f> deltaCapture;
global_variable vector<vector<Vector2f>> samples;
global_variable vector<vector<Vector2f>> deltas;
global_variable vector<Vector2f> labels;
global_variable bool isCapturingEight = false;

global_variable float *verify;
global_variable float *verifyLabel;

float mouseX = WINHALFWIDTH;
float mouseY = WINHALFHEIGHT;

Net neural;						
Buffer backBuffer;

void RecordSample(float label){
	vector<Vector2f> mods;
	mods.push_back(Vector2f(1.f, 1.f));
	mods.push_back(Vector2f(-1.f, 1.f));
	mods.push_back(Vector2f(1.f, -1.f));
	mods.push_back(Vector2f(-1.f, -1.f));
	vector<Vector2f> sample = mouseCapture;
	vector<Vector2f> delta = deltaCapture;
	for (int c = 0; c < mods.size(); c++) {
		vector<Vector2f> tempSample = sample;
		vector<Vector2f> tempDelta = delta;
		for( int i = 0; i < MAXCAPTURECOUNT; i++ ) {
			tempSample[i] = Vector2f(tempSample[i][0] * mods[c][0] + ( mods[c][0] > 0 ? 0 : WINHEIGHT),
									 tempSample[i][1] * mods[c][1] + ( mods[c][1] > 0 ? 0 : WINWIDTH ));
			tempDelta[i] = Vector2f(tempDelta[i][0] * mods[c][0], tempDelta[i][1] * mods[c][1] );
		}
		samples.push_back(tempSample);
		deltas.push_back(tempDelta);
		labels.push_back(Vector2f(label, 0.f));
	}
}

internal LRESULT CALLBACK Win32MainWindowCallback(HWND Window, UINT Message, WPARAM WParam, LPARAM LParam) {
	LRESULT Result = 0;
	switch(Message) {
	case WM_DESTROY:
	case WM_CLOSE:
	{
		globalRunning = false;
	} break;
	case WM_MOUSEMOVE:
	{
		mouseX = float(GET_X_LPARAM(LParam));
		mouseY = float(GET_Y_LPARAM(LParam)) + backBuffer.titleOffset - 25;
	} //fall through
	case WM_KEYUP:
	{
		switch( WParam ) {
			case '8':
			isCapturingEight = false;
			samples.erase(samples.end() - MAXCAPTURECOUNT, samples.end());
			deltas.erase(deltas.end() - MAXCAPTURECOUNT, deltas.end());
			labels.erase(labels.end() - MAXCAPTURECOUNT, labels.end());
			RecordSample(1.f);
			break; 
			case 'V': //fall through
			*verify = 1.f;
			skipStep = true;
			break;
			case 'C':
			*verifyLabel = 0.f;
			skipStep = true;
			break;
		}
	
	} break;
	case WM_KEYDOWN:
	{
		switch(WParam) {
			case '8':
			isCapturingEight = true;
			break;
		default:
			break;
		}
	}
	case WM_ACTIVATEAPP:
	{
		OutputDebugStringA("WM_ACTIVATEAPP\n");
	} break;
	default:
	{
		Result = DefWindowProcA(Window, Message, WParam, LParam);
	} break;
	}
	return Result;
}

 void UpdateDisplay( vector<Vector2f> &mouseTrail, vector<Vector2f> &mouseCapture) {
	 if( globalRunning ) {
		ClearScreen(backBuffer);
		 		
		for (int mT =1; mT < mouseTrail.size() - 1; mT++) {
			DrawLine(backBuffer, mouseTrail[mT-1][0], float(WINHEIGHT) - mouseTrail[mT-1][1],
					mouseTrail[mT][0], float(WINHEIGHT) - mouseTrail[mT][1], Color(200, 200, 200, 0) * ( float(mT) / float(MOUSETRAILLENGTH)) );
		}
		if( mouseCapture.size() ) {
			for( int mC = 0; mC < mouseCapture.size(); mC++ ) {
				DrawFilledCircle(backBuffer, mouseCapture[mC][0], float(WINHEIGHT) - mouseCapture[mC][1], 10.f, isCapturingEight ? Color(0, 200, 0, 0) : Color(200, 200, 200, 0));
			}
		}
	}	
}

void UpdateWinTitle(int &steps, HWND window) {
	char s[255];
	sprintf_s(s, "%d", steps);
	SetWindowText(window, LPCSTR(s));
}

MatrixXf CreateSparseData(int pointCount) {
	float piAdjusted = Pi32 - 0.01f;
	float delta = (2.f * Pi32) / float(pointCount);
	MatrixXf out = MatrixXf(2, pointCount+1);
	for(int i = 0; i <= pointCount; i++) {
		float r = clamp(-Pi32 + (delta * i), -piAdjusted, piAdjusted);
		out(0,i) = sin(r);
		out(1,i) = cos(r);
	}
	return out;
}

void UpdateHistory(vector<float> &history, float cost) {
	history.push_back(min((cost) * WINHEIGHT, WINHEIGHT));
	if(history.size() >= WINWIDTH + WINWIDTH) {
		for(int i = 1; i < (int)history.size(); i += 2) {
			history.erase(history.begin() + i);
		}
	}
}

void CopyVecToMatrix(MatrixXf &dest, vector<vector<Vector2f>> orig) {
	int k = 0;
	for( int i = 0; i < orig.size() - 1; i++ ) {
		for( int j = 0; j < MAXCAPTURECOUNT; j++ ) {
			*( dest.data() + k++ ) = float(orig[i][j][0]);
			*( dest.data() + k++ ) = float(orig[i][j][1]);
		}
	}
}

void ContainVector(vector<Vector2f> &vec, int maxSize) {
	if( vec.size() > maxSize ) {
		vec.erase(vec.begin());
	}
}

void CopyNestedVec(MatrixXf &saveData, vector<vector<Vector2f>> * s, int maxCount) {
	int k = 0;
	for( int i = 0; i < saveData.cols(); i++ ) {
		for( int j = 0; j < maxCount; j++ ) {
			*( saveData.data() + k++ ) = float(( *s )[i][j][0]);
			*( saveData.data() + k++ ) = float(( *s )[i][j][1]);
		}
	}
}

int CALLBACK WinMain(HINSTANCE Instance, HINSTANCE PrevInstance, LPSTR CommandLine, int ShowCode) {
	MatrixXf readSamples = BuildMatFromFile("GestureSamples.csv").transpose();
	MatrixXf readDeltas = BuildMatFromFile("GestureDeltas.csv").transpose();
	MatrixXf readLabels = BuildMatFromFile("GestureLabels.csv").transpose();
	assert(readSamples.size() == readDeltas.size());

	WNDCLASSA winClass = {};
	InitializeWindow(&winClass, Instance, Win32MainWindowCallback, &backBuffer, WINWIDTH, WINHEIGHT, "NN_PredictRadian");

	if(RegisterClassA(&winClass)) {
		HWND window = CreateWindowExA(0, winClass.lpszClassName, "NNet||",
									  WS_OVERLAPPED | WS_SYSMENU |WS_VISIBLE, CW_USEDEFAULT, CW_USEDEFAULT,
									  WINWIDTH, WINHEIGHT, 0, 0, Instance, 0);
		
		HDC deviceContext = GetDC(window);
		vector<Vector2f> mouseTrail;

		int steps = 0;
		//Main Loop
		while(globalRunning) {
			Win32ProcessPendingMessages();	
			if( isVerifying){
				if( steps < readLabels.cols() && readLabels.cols() > 0 ) {
					verifyLabel = &readLabels(0, steps);
					verify = &readLabels(1, steps);
					if( !skipStep && readLabels(0, steps) == 1.f && readLabels(1, steps) == 0.f ) {
						for( int i = 0; i < readSamples.rows(); ) {
							float valX = *( readSamples.data() + ( steps * readSamples.rows() + i++ ) );
							float valY = *( readSamples.data() + ( steps * readSamples.rows() + i++ ) );
							DrawFilledCircle(backBuffer, valX, WINHEIGHT - valY, 10.f, Color(i * 10, 200, 0, 0));
						}
					} else {
						skipStep = false;
						ClearScreen(backBuffer);
						steps++;
					}
					Win32DisplayBufferInWindow(deviceContext, window, backBuffer);
					continue;
				}
				globalRunning = false;
			}
			if( mouseCapture.size() == 0 ) {
				mouseCapture.push_back(Vector2f(float(mouseX), float(mouseY)));
			}
			ContainVector(mouseTrail, MOUSETRAILLENGTH);
			mouseTrail.push_back(Vector2f(float(mouseX), float(mouseY)));
			Vector2f end = mouseTrail.back();
			Vector2f delta = mouseCapture.back() - end;
			float dist = sqrtf(delta[0] * delta[0] + delta[1] * delta[1]);
			if( dist >= CAPTURETHRESHOLD ){
				deltaCapture.push_back(delta/CAPTURETHRESHOLD);
				mouseCapture.push_back(end);
				ContainVector(mouseCapture, MAXCAPTURECOUNT);
				ContainVector(deltaCapture, MAXCAPTURECOUNT);
				if( !isCapturingEight && deltaCapture.size() == MAXCAPTURECOUNT) {
					assert(deltaCapture.size() == mouseCapture.size());
					RecordSample(0.f);
				}
			}

			UpdateDisplay(mouseTrail, mouseCapture);
			Win32DisplayBufferInWindow(deviceContext, window, backBuffer);
			UpdateWinTitle(steps, window);
		}

		if( !isVerifying ) {
			MatrixXf saveSamples = MatrixXf(MAXCAPTURECOUNT * 2, samples.size() - 1);
			MatrixXf saveDeltas = MatrixXf(MAXCAPTURECOUNT * 2, deltas.size() - 1);
			MatrixXf saveLabels = MatrixXf(2,samples.size() - 1);
			CopyNestedVec(saveSamples, &samples, MAXCAPTURECOUNT);
			CopyNestedVec(saveDeltas, &deltas, MAXCAPTURECOUNT);
			int l = 0;
			for( int i = 0; i < samples.size()-1; i++ ) {
				*( saveLabels.data() + l++ ) = float(labels[i][0]);
				*( saveLabels.data() + l++ ) = float(labels[i][1]);
			}
			if( readSamples.size() + readLabels.size() + readDeltas.size() > 0 ) {
				MatrixXf outSamples = MatrixXf(readSamples.rows(), readSamples.cols() + saveSamples.cols());
				outSamples << readSamples, saveSamples;
				MatrixXf outDeltas = MatrixXf(readDeltas.rows(), readDeltas.cols() + saveDeltas.cols());
				outSamples << readDeltas, saveDeltas;
				MatrixXf outLabels = MatrixXf(readLabels.rows(), readLabels.cols() + saveLabels.cols());
				outLabels << readLabels, saveLabels;
				writeToCSVfile("GestureSamples.csv", outSamples.transpose());
				writeToCSVfile("GestureDeltas.csv", outDeltas.transpose());
				writeToCSVfile("GestureLabels.csv", outLabels.transpose());
			}
			else {
				writeToCSVfile("GestureSamples.csv", saveSamples.transpose());
				writeToCSVfile("GestureDeltas.csv", saveDeltas.transpose());
				writeToCSVfile("GestureLabels.csv", saveLabels.transpose());
			}
		} else {
			writeToCSVfile("GestureLabels.csv", readLabels.transpose());
		}
		DeleteDC(deviceContext);
	}
	return EXIT_SUCCESS;
}
