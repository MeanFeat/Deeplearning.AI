#include "win32_ExpertSystems.h"
#include "stdMat.h"
#include "windowsx.h"
#include <time.h>

#define WINWIDTH 30
#define WINHEIGHT WINWIDTH
#define WINHALFWIDTH int((WINWIDTH-1)*0.5f)
#define WINHALFHEIGHT WINHALFWIDTH
#define PLOTDIST 120.f

global_variable bool globalRunning = true;
global_variable bool isTraining = true;
global_variable bool drawOutput = false;
global_variable bool plotData = false;

static time_t startTime;
static time_t currentTime;


Buffer backBuffer;
Net neural;
NetTrainer trainer;

internal LRESULT CALLBACK Win32MainWindowCallback(HWND Window, UINT Message, WPARAM WParam, LPARAM LParam) {
	LRESULT Result = 0;
	switch(Message) {
	case WM_DESTROY:
	case WM_CLOSE:
	{
		globalRunning = false;
	} break;
	case WM_KEYDOWN:
	{
		switch(WParam) {
		case 'T':
			isTraining = !isTraining;
			break;
		case 'D':
			drawOutput = !drawOutput;
			break;
		case 'P':
			plotData = !plotData;
			break;
		case 'W':
			trainer.ModifyLearningRate(0.02f);
			break;
		case 'S':
			trainer.ModifyLearningRate(-0.02f);
			break;
		case 'Q':
			trainer.ModifyRegTerm(0.02f);
			break;
		case 'A':
			trainer.ModifyRegTerm(-0.02f);
			break;
		default:
			break;
		}
	} break;
	case WM_LBUTTONDOWN: //fall through
	case WM_MOUSEMOVE:
	{
	} break;

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

MatrixXf BuildGrey(MatrixXf m) {
	return MatrixXf::Ones(m.rows(), m.cols()) * 0.5f;
}

MatrixXf CreateData(int n) {
	return MatrixXf::Random(WINWIDTH*WINHEIGHT, n);
}

void DrawOutputToScreen() {
	MatrixXf h = neural.ForwardPropagation(CreateData(1));
	int *pixel = (int *)backBuffer.memory;
	for(int i = 0; i < h.rows(); i++) {
		int val = int(h(i, 0) * 255);
		*pixel++ = Color(val, val, val, 255).ToBit();
	}
}

void UpdateDisplay( MatrixXf X, MatrixXf Y, vector<float> &history, vector<float> &testHistory) {
	if(drawOutput) {
		DrawOutputToScreen();
	} else {
		ClearScreen(backBuffer);
	}
	if(plotData) {
	}
	DrawHistory(backBuffer, history, Color(200, 90, 90, 255));
	
}

void UpdateWinTitle(int &steps, HWND window) {
	time(&currentTime);
	char s[255];
	sprintf_s(s, "%d|T:%0.f|C:%0.10f|LR:%0.2f|RT:%0.2f|"
			  , steps,difftime(currentTime, startTime), trainer.GetCache().cost, trainer.GetTrainParams().learningRate, trainer.GetTrainParams().regTerm);

	SetWindowText(window, LPCSTR(s));
}


void UpdateHistory(vector<float> &history, float cost) {
	history.push_back(min((cost) * WINHEIGHT, WINHEIGHT));
	if(history.size() >= WINWIDTH + WINWIDTH) {
		for(int i = 1; i < (int)history.size(); i += 2) {
			history.erase(history.begin() + i);
		}
	}
}


int CALLBACK WinMain(HINSTANCE Instance, HINSTANCE PrevInstance, LPSTR CommandLine, int ShowCode) {
	MatrixXf X;
	MatrixXf testX;
	MatrixXf Y;
	MatrixXf testY;
	WNDCLASSA winClass = {};
	InitializeWindow(&winClass, Instance, Win32MainWindowCallback, &backBuffer, WINWIDTH, WINHEIGHT, "NN_PredictRadian");
	
	X = CreateData(10000);
	Y = BuildGrey(X);	

	time(&startTime);
	if(RegisterClassA(&winClass)) {
		HWND window = CreateWindowExA(0, winClass.lpszClassName, "NNet||",
									  WS_OVERLAPPED | WS_SYSMENU |WS_VISIBLE, CW_USEDEFAULT, CW_USEDEFAULT,
									  WINWIDTH*8,(WINHEIGHT*8), 0, 0, Instance, 0);
		int bredth = int(X.rows());
		neural = Net(bredth, { bredth,bredth,bredth }, bredth, { ReLU,ReLU,ReLU, Sigmoid });
		trainer = NetTrainer(&neural, &X, &Y, 0.15f, 0.125f, 3.f);

		HDC deviceContext = GetDC(window);
		vector<float> history;
		vector<float> testHistory;

		int steps = 0;
		//Main Loop
		while(globalRunning) {
			Win32ProcessPendingMessages();	
			if(isTraining) {
				trainer.UpdateSingleStep();
				UpdateHistory(history, trainer.GetCache().cost);
				//UpdateHistory(testHistory, trainer.CalcCost(neural.ForwardPropagation(testX), testY));
				steps++;
			} else {
			}
			UpdateDisplay( X, Y, history, testHistory);
			Win32DisplayBufferInWindow(deviceContext, window, backBuffer);			
			UpdateWinTitle(steps, window);
		}
		DeleteDC(deviceContext);
	}
	return EXIT_SUCCESS;
}
