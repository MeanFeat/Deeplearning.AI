#include "win32_ExpertSystems.h"
#include "stdMatrix.h"
#include "windowsx.h"
#include <time.h>

#define WINWIDTH 350
#define WINHEIGHT WINWIDTH
#define WINHALFWIDTH int((WINWIDTH-1)*0.5f)
#define WINHALFHEIGHT WINHALFWIDTH
#define PLOTDIST 120.f

global_variable bool globalRunning = true;
global_variable bool isTraining = true;
global_variable bool drawOutput = true;
global_variable bool plotData = false;

global_variable vector<double> predictions;
static time_t startTime;
static time_t currentTime;

double mouseX = WINHALFWIDTH;
double mouseY = WINHALFHEIGHT + 100;

Buffer backBuffer;
Net neural;
d_NetTrainer trainer;

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
		if(DWORD(WParam) & MK_LBUTTON) {
			mouseX = double(GET_X_LPARAM(LParam));
			mouseY = double(GET_Y_LPARAM(LParam)) + backBuffer.titleOffset;
		}
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

MatrixXd NormalizeData(MatrixXd m) {
	MatrixXd a = (m.row(0).array().pow(2) + m.row(1).array().pow(2)).array().sqrt();
	MatrixXd b = MatrixXd(2, a.cols());
	b << a, a;
	return MatrixXd(m.array() / b.array());
}

Color GetColorBlend(double percent) {
	return percent < 0.f ? Color(255, 255, 255, 255).Blend(negativeColor, -percent)
		: Color(255, 255, 255, 255).Blend(positiveColor, percent);;
}

void PlotData(MatrixXd X, MatrixXd Y) {
	for(int i = 0; i < X.cols(); ++i) {
		DrawFilledCircle(backBuffer, int(WINHALFWIDTH + X(0, i)), int(WINHALFHEIGHT + -X(1, i)), 10.f, GetColorBlend(Y(0, i)));
	}
}

void DrawOutputToScreen(MatrixXd normScreen) {
	trainer.Visualization(normScreen, (int*)backBuffer.memory, backBuffer.width, backBuffer.height, false);
}

void UpdateDisplay(MatrixXd screenCoords, MatrixXd X, MatrixXd Y, vector<double> &history, vector<double> &testHistory) {
	if(globalRunning) {
		if(drawOutput) {
			DrawOutputToScreen(screenCoords);
		} else {
			ClearScreen(backBuffer);
		}
		if (plotData){

			PlotData(X*PLOTDIST, Y);
		}

		DrawLine(backBuffer, WINHALFWIDTH, WINHALFHEIGHT, mouseX, double(WINHEIGHT - mouseY), Color(0, 0, 255, 255));
		for (int i = 0; i < (int)predictions.size();++i){
			int predX = int(sin(predictions[i] * Pi32) * 100.f);
			int predY = int(cos(predictions[i] * Pi32) * 100.f);
			DrawLine(backBuffer, WINHALFWIDTH, WINHALFHEIGHT, double(WINHALFWIDTH) + predX, double(WINHALFHEIGHT) - predY, Color(100, 0, 0, 0));
		}
		DrawHistory(backBuffer, history, Color(200, 90, 90, 255));
		DrawHistory(backBuffer, testHistory, Color(100, 100, 100, 255));
	}	
}

void UpdateWinTitle(int &steps, HWND window) {
	time(&currentTime);
	char s[255];
	sprintf_s(s, "%d|T:%0.f|C:%0.10f|LR:%0.2f|RT:%0.2f|"
			  , steps,difftime(currentTime, startTime), trainer.GetCache().cost, trainer.GetTrainParams().learningRate, trainer.GetTrainParams().regTerm);
	char r[255];	
	sprintf_s(r, " |%0.2f|%0.2f| ", atan2((mouseX - WINHALFWIDTH), (mouseY - WINHALFHEIGHT)), predictions[0]*Pi32);
	strcat_s(s, r);
	SetWindowText(window, LPCSTR(s));
}

MatrixXd BuildRadians(MatrixXd m) {
	double coefficient = 1.f / Pi32;
	MatrixXd out = MatrixXd(1, m.cols());
	for(int i = 0; i < m.cols(); ++i) {
		out(0, i) = atan2((m(0, i)), (m(1, i))) * coefficient;
		//out(1, i) = atan2(-(m(1, i)), (m(0, i))) * coefficient;
		//out(2, i) = atan2((m(1, i)), -(m(0, i))) * coefficient;
	}
	return out;
}

void UpdatePrediction() {
	trainer.UpdateNetwork();
	MatrixXd mouse = MatrixXd(2,1);
	mouse(0, 0) = mouseX-WINHALFWIDTH;
	mouse(1, 0) = mouseY-WINHALFHEIGHT;
	MatrixXd h = neural.ForwardPropagation(NormalizeData(mouse));
	predictions.clear();
	for (int i = 0; i < h.rows(); ++i){
		predictions.push_back(h(i, 0));
	}
}

MatrixXd CreateSparseData(int pointCount) {
	double piAdjusted = Pi32 - 0.01f;
	double delta = (2.f * Pi32) / double(pointCount);
	MatrixXd out = MatrixXd(2, pointCount+1);
	for(int i = 0; i <= pointCount; i++) {
		double r = clamp(-Pi32 + (delta * i), -piAdjusted, piAdjusted);
		out(0,i) = sin(r);
		out(1,i) = cos(r);
	}
	return out;
}

void UpdateHistory(vector<double> &history, double cost) {
	history.push_back(min((cost) * WINHEIGHT, WINHEIGHT));
	if(history.size() >= WINWIDTH + WINWIDTH) {
		for(int i = 1; i < (int)history.size(); i += 2) {
			history.erase(history.begin() + i);
		}
	}
}

int CALLBACK WinMain(HINSTANCE Instance, HINSTANCE PrevInstance, LPSTR CommandLine, int ShowCode) {
	MatrixXd X;
	MatrixXd testX;
	MatrixXd Y;
	MatrixXd testY;
	WNDCLASSA winClass = {};
	InitializeWindow(&winClass, Instance, Win32MainWindowCallback, &backBuffer, WINWIDTH, WINHEIGHT, "NN_PredictRadian");
	MatrixXd screenCoords = NormalizeData(BuildDisplayCoords(backBuffer).transpose());

	testX = NormalizeData(CreateSparseData(99));
	testY = BuildRadians(testX);
	X = NormalizeData(CreateSparseData(45));
	Y = BuildRadians(X);	

	time(&startTime);
	if(RegisterClassA(&winClass)) {
		HWND window = CreateWindowExA(0, winClass.lpszClassName, "NNet||",
									  WS_OVERLAPPED | WS_SYSMENU |WS_VISIBLE, CW_USEDEFAULT, CW_USEDEFAULT,
									  WINWIDTH, WINHEIGHT, 0, 0, Instance, 0);
		
		neural = Net((int)X.rows(), { 8, 8 }, (int)Y.rows(), { Tanh, Tanh, Tanh });
		trainer = d_NetTrainer(&neural, &X, &Y, 0.5, 0.25, 1.0);

		HDC deviceContext = GetDC(window);
		vector<double> history;
		vector<double> testHistory;

		int steps = 0;
		//Main Loop
		while(globalRunning) {
			Win32ProcessPendingMessages();	
			if (isTraining) {
				trainer.UpdateSingleStep();
				UpdateHistory(history, trainer.GetCache().cost);
				UpdateHistory(testHistory, trainer.CalcCost(neural.ForwardPropagation(testX), testY));
				steps++;
			} else {
			}
			UpdateDisplay(screenCoords, X, Y, history, testHistory);
			UpdatePrediction();
			Win32DisplayBufferInWindow(deviceContext, window, backBuffer);
			
			UpdateWinTitle(steps, window);
		}
		DeleteDC(deviceContext);
	}
	return EXIT_SUCCESS;
}
