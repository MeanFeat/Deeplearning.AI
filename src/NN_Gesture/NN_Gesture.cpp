#include "win32_ExpertSystems.h"
#include "stdMat.h"
#include "windowsx.h"
#include <time.h>

#define WINWIDTH 800
#define WINHEIGHT WINWIDTH
#define WINHALFWIDTH int((WINWIDTH-1)*0.5f)
#define WINHALFHEIGHT WINHALFWIDTH
#define MAXCAPTURECOUNT 25
#define MOUSETRAILLENGTH 500
#define CAPTURETHRESHOLD 40.f

global_variable bool globalRunning = true;
global_variable bool isTraining = false;
global_variable bool isVerifying = false;
global_variable bool isRecordingData = false;
global_variable bool shouldSaveChanges = false;
global_variable bool skipStep = false;

global_variable vector<Vector2f> mouseCapture;
global_variable vector<Vector2f> lastSuccesful8;
global_variable vector<Vector2f> lastSuccesful8Deltas;
global_variable vector<Vector2f> deltaCapture;
global_variable vector<vector<Vector2f>> samples;
global_variable vector<vector<Vector2f>> deltas;
global_variable vector<float> labels;
global_variable bool isCapturingEight = false;

global_variable float *verify;
global_variable float *verifyLabel;

float mouseX = WINHALFWIDTH;
float mouseY = WINHALFHEIGHT;

Net neural;
NetTrainer trainer;
Buffer backBuffer;

void RecordSample(vector<Vector2f> vec, float label){
	vector<Vector2f> mods;
	mods.push_back(Vector2f(1.f, 1.f));
	mods.push_back(Vector2f(-1.f, 1.f));
	mods.push_back(Vector2f(1.f, -1.f));
	mods.push_back(Vector2f(-1.f, -1.f));
	vector<Vector2f> sample = mouseCapture;
	vector<Vector2f> delta = deltaCapture;
	for( int c = 0; c < mods.size(); c++ ) {
		vector<Vector2f> tempSample = sample;
		vector<Vector2f> tempDelta = delta;
		for( int i = 0; i < MAXCAPTURECOUNT; i++ ) {
			tempSample[i] = Vector2f(tempSample[i][0] * mods[c][0] + ( mods[c][0] > 0 ? 0 : WINHEIGHT ),
									 tempSample[i][1] * mods[c][1] + ( mods[c][1] > 0 ? 0 : WINWIDTH ));
			tempDelta[i] = Vector2f(tempDelta[i][0] * mods[c][0], tempDelta[i][1] * mods[c][1]);
		}
		samples.push_back(tempSample);
		deltas.push_back(tempDelta);
		labels.push_back(label);
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
			RecordSample(deltaCapture, 1.f);
			break; 
			case 'V': //fall through
			*verifyLabel = 1.f;
			skipStep = true;
			break;
			case 'C':
			*verifyLabel = 0.f;
			skipStep = true;
			case 'T':
			isTraining = !isTraining;
			break;
			case '0':
			RecordSample(deltaCapture, 0.f);
			shouldSaveChanges = true;
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
		ShowCursor(false);
		OutputDebugStringA("WM_ACTIVATEAPP\n");
	} break;
	default:
	{
		Result = DefWindowProcA(Window, Message, WParam, LParam);
	} break;
	}
	return Result;
}
#define ICONLEFT 50
#define ICONTOP	100

global_variable float successFade = 0.f;
 void UpdateDisplay( vector<Vector2f> &mTrail, vector<Vector2f> &mCap, vector<float> &hist, float h) {
	 if( globalRunning ) {
		ClearScreen(backBuffer);
		if (isTraining) {
			DrawHistory(backBuffer, hist, Color(200, 100, 100, 255));
		} else {
			for (int i = -3; i < 3; i++) {
				 if (successFade > 0.f){
					 DrawLine(backBuffer, ICONLEFT, ICONTOP + i, ICONLEFT - 25, ICONTOP + i + 25, Color(0, 255 * successFade, 0, 0));
					 DrawLine(backBuffer, ICONLEFT, ICONTOP + i, ICONLEFT + 75, ICONTOP + i + 75, Color(0, 255 * successFade, 0, 0));
				 } else {
					 DrawLine(backBuffer, ICONLEFT + 50, ICONTOP + i - 50, ICONLEFT - 50, ICONTOP + i + 50, Color(200, 0, 0, 0));
					 DrawLine(backBuffer, ICONLEFT - 50, ICONTOP + i - 50, ICONLEFT + 50, ICONTOP + i + 50, Color(200, 0, 0, 0));
				 }
			}

			for( int mT = 1; mT < mTrail.size() - 1; mT++ ) {
				DrawLine(backBuffer, mTrail[mT - 1][0], float(WINHEIGHT) - mTrail[mT - 1][1],
						 mTrail[mT][0], float(WINHEIGHT) - mTrail[mT][1], Color(200, 200, 200, 0) * ( float(mT) / float(MOUSETRAILLENGTH) ));
			}
			if( isRecordingData && mCap.size() ) {
				for( int mC = 0; mC < mCap.size(); mC++ ) {
					DrawFilledCircle(backBuffer, mCap[mC][0], float(WINHEIGHT) - mCap[mC][1], 10.f, isCapturingEight ? Color(0, 200, 0, 0) : Color(200, 200, 200, 0));
				}
			}
			if (h > 0.9f){
				lastSuccesful8.clear();
				lastSuccesful8Deltas.clear();
				lastSuccesful8 = mouseCapture;
				lastSuccesful8Deltas = deltaCapture;
				successFade = 1.f;
			}
			if( lastSuccesful8.size() > 0 ) {
				for( int ls = 0; ls < lastSuccesful8.size(); ls++ ) {
					DrawFilledCircle(backBuffer, lastSuccesful8[ls][0], float(WINHEIGHT) - lastSuccesful8[ls][1], 10.f, Color(0, ls * 10 * successFade, 0, 0));
				}
			}
		}
		successFade = max(0.f,successFade - 0.001f);
		DrawFilledCircle(backBuffer,mouseX, float(WINHEIGHT) - mouseY, 5.f, Color(200, 200, 200, 0));
	}	
}

void UpdateWinTitle(int &steps,float &prediciton, HWND window) {
	char s[255];
	sprintf_s(s, "%d ||Cost: %0.10f || Is it an 8: %0.10f ", steps,trainer.GetCache().cost, prediciton);
	char prompt[255];
	if (successFade > 0.f){
		sprintf_s(prompt, "Great!");
	} else {
		sprintf_s(prompt, "hmmm... Draw a figure eight.");
	}
	strcat_s(s, prompt);
	SetWindowText(window, LPCSTR(s));
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

void UpdateHistory(vector<float> &history) {
	history.push_back(min(( trainer.GetCache().cost ) * ( WINHEIGHT - backBuffer.titleOffset ), WINHEIGHT));
	if( history.size() >= WINWIDTH + WINWIDTH ) {
		for( int i = 1; i < (int)history.size(); i += 2 ) {
			history.erase(history.begin() + i);
		}
	}
}

int CALLBACK WinMain(HINSTANCE Instance, HINSTANCE PrevInstance, LPSTR CommandLine, int ShowCode) {
	MatrixXf readSamples = BuildMatFromFile("GestureSamplesTrain.csv").transpose();
	MatrixXf readDeltas = BuildMatFromFile("GestureDeltasTrain.csv").transpose();
	MatrixXf readLabels = BuildMatFromFile("GestureLabelsTrain.csv").transpose();

	WNDCLASSA winClass = {};
	InitializeWindow(&winClass, Instance, Win32MainWindowCallback, &backBuffer, WINWIDTH, WINHEIGHT, "NN_PredictRadian");

	if(RegisterClassA(&winClass)) {
		HWND window = CreateWindowExA(0, winClass.lpszClassName, "NNet||",
									  WS_OVERLAPPED | WS_SYSMENU |WS_VISIBLE, CW_USEDEFAULT, CW_USEDEFAULT,
									  WINWIDTH, WINHEIGHT, 0, 0, Instance, 0);
		
		HDC deviceContext = GetDC(window);
		vector<Vector2f> mouseTrail;
		int sampleIndex = 0;
		neural = Net("Gesture-Weights.json");
		/*neural = Net((int)readDeltas.rows(), {80,80}, (int)readLabels.rows(), {
			Tanh,
			Tanh,
			Sigmoid});*/
		
		trainer = NetTrainer(&neural, &readDeltas, &readLabels, 0.15f, 1.25f, 20.f);
		vector<float> history;
		float h = 0.f;
		int steps = 0;
		//Main Loop
		while(globalRunning) {
			if( isTraining) {
				for (int e = 0; e < 50; e++) {
					trainer.UpdateSingleStep();
				}
				UpdateHistory(history);
			}

			Win32ProcessPendingMessages();
		 
			if( isVerifying){
				if( steps < readLabels.cols() && readLabels.cols() > 0 ) {
					verifyLabel = &readLabels(0, steps);
					//verify = &readLabels(1, steps);
					if( !skipStep){// && readLabels(0, steps) == 0.f){// && readLabels(1, steps) == 0.f ) {
						for( int i = 0; i < readSamples.rows(); ) {
							float valX = *( readSamples.data() + ( steps * readSamples.rows() + i++ ) );
							float valY = *( readSamples.data() + ( steps * readSamples.rows() + i++ ) );
							DrawFilledCircle(backBuffer, valX, WINHEIGHT - valY, 10.f, Color(i * 10, 200, 0, 0));
						}
					} else {
						skipStep = false;
						ClearScreen(backBuffer);
						steps+=4;
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
					if (sampleIndex++ > MAXCAPTURECOUNT) {
						RecordSample(deltaCapture, 0.f);
						sampleIndex = 0;
					}
					if( !isRecordingData ) {
						MatrixXf X = MatrixXf(MAXCAPTURECOUNT * 2, 1);
						int l = 0;
						for( int i = 0; i < deltaCapture.size(); i++ ) {
							float xVal = float(deltaCapture[i][0]);
							float yVal = float(deltaCapture[i][1]);
							*( X.data() + l++ ) = xVal;
							*( X.data() + l++ ) = yVal;
						}
						MatrixXf output = neural.ForwardPropagation(X);
						h = output(0, 0);
					}
				}
			}
			UpdateDisplay(mouseTrail, mouseCapture, history, h);
			Win32DisplayBufferInWindow(deviceContext, window, backBuffer);
			UpdateWinTitle(steps, h, window);
		}

		if ((!isTraining && isRecordingData) || shouldSaveChanges ) {
			if( !isVerifying && samples.size() > MAXCAPTURECOUNT ) {
				MatrixXf saveSamples = MatrixXf(MAXCAPTURECOUNT * 2, samples.size() - 1);
				MatrixXf saveDeltas = MatrixXf(MAXCAPTURECOUNT * 2, deltas.size() - 1);
				MatrixXf saveLabels = MatrixXf(1, samples.size() - 1);
				CopyNestedVec(saveSamples, &samples, MAXCAPTURECOUNT);
				CopyNestedVec(saveDeltas, &deltas, MAXCAPTURECOUNT);
				int l = 0;
				for( int i = 0; i < samples.size() - 1; i++ ) {
					*( saveLabels.data() + l++ ) = float(labels[i]);
				}
				if( readSamples.size() + readLabels.size() + readDeltas.size() > 0 ) {
					MatrixXf outSamples = MatrixXf(readSamples.rows(), readSamples.cols() + saveSamples.cols());
					outSamples << readSamples, saveSamples;
					MatrixXf outDeltas = MatrixXf(readDeltas.rows(), readDeltas.cols() + saveDeltas.cols());
					outSamples << readDeltas, saveDeltas;
					MatrixXf outLabels = MatrixXf(readLabels.rows(), readLabels.cols() + saveLabels.cols());
					outLabels << readLabels, saveLabels;
					writeToCSVfile("GestureSamplesTrain.csv", outSamples.transpose());
					writeToCSVfile("GestureDeltasTrain.csv", outDeltas.transpose());
					writeToCSVfile("GestureLabelsTrain.csv", outLabels.transpose());
				} else {
					writeToCSVfile("GestureSamplesTrain.csv", saveSamples.transpose());
					writeToCSVfile("GestureDeltasTrain.csv", saveDeltas.transpose());
					writeToCSVfile("GestureLabelsTrain.csv", saveLabels.transpose());
				}
			} else {
				writeToCSVfile("GestureLabelsTrain.csv", readLabels.transpose());
			}
		}
		DeleteDC(deviceContext);
	}
	return EXIT_SUCCESS;
}
