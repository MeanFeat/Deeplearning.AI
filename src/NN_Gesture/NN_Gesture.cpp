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

float mouseX = WINHALFWIDTH;
float mouseY = WINHALFHEIGHT;

Net neural;
Buffer backBuffer;

void RecordSample(float label){
	vector<Vector2f> sample = mouseCapture;
	vector<Vector2f> delta = deltaCapture;
	samples.push_back(sample);
	deltas.push_back(delta);
	labels.push_back(Vector2f(label, 0.f));
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
			RecordSample(1.f);
			break; 
			case 'V': //fall through
			*verify = 1.f;
			case 'C':
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

int CALLBACK WinMain(HINSTANCE Instance, HINSTANCE PrevInstance, LPSTR CommandLine, int ShowCode) {
	MatrixXf readSamples = BuildMatFromFile("GestureData.csv");
	MatrixXf readDeltas = BuildMatFromFile("GestrureDeltas.csv");
	MatrixXf readLabels = BuildMatFromFile("GestureLabels.csv");

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
			if( isVerifying && steps < readLabels.cols() && readLabels.cols() > 0 ) {
				verify = &readLabels(1, steps);
				if(!skipStep && readLabels(0, steps) == 1.f && readLabels(1, steps) == 0.f) {
					for( int i = 0; i < readSamples.rows(); ) {
						float valX = *( readSamples.data() + ( steps * readSamples.rows() + i++ ) );
						float valY = *( readSamples.data() + ( steps * readSamples.rows() + i++ ) );						 
						DrawFilledCircle(backBuffer, valX, WINHEIGHT - valY, 10.f, Color(100, 200, 0, 0));
					}
				} else {
					skipStep = false;
					ClearScreen(backBuffer);
					steps++;
				}
				Win32DisplayBufferInWindow(deviceContext, window, backBuffer);
				continue;
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
				deltaCapture.push_back(delta);
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
			MatrixXf saveData = MatrixXf(MAXCAPTURECOUNT * 2, samples.size() - 1);
			MatrixXf saveLabels = MatrixXf(2,samples.size() - 1);
			int k = 0;
			for( int i = 0; i < saveData.cols(); i++ ) {
				for( int j = 0; j < MAXCAPTURECOUNT; j++ ) {
					*( saveData.data() + k++ ) = float(samples[i][j][0]);
					*( saveData.data() + k++ ) = float(samples[i][j][1]);
				}
			}
			int l = 0;
			for( int i = 0; i < samples.size()-1; i++ ) {
				*( saveLabels.data() + l++ ) = float(labels[i][0]);
				*( saveLabels.data() + l++ ) = float(labels[i][1]);
			}
			if (readSamples.size() > 0 && readLabels.size() > 0) {
				MatrixXf outData = MatrixXf(readSamples.rows() + saveData.rows(), saveData.cols());
				outData << readSamples, saveData;
				MatrixXf outLabels = MatrixXf(readLabels.rows() + saveLabels.rows(), saveLabels.cols());
				outLabels << readLabels, saveLabels;
				writeToCSVfile("GestureData.csv", outData);
				writeToCSVfile("GestureLabels.csv", outLabels);
			}
			else {
				writeToCSVfile("GestureData.csv", saveData);
				writeToCSVfile("GestureLabels.csv", saveLabels);
			}
		} else {
			writeToCSVfile("GestureLabels.csv", readLabels);
		}
		DeleteDC(deviceContext);
	}
	return EXIT_SUCCESS;
}
