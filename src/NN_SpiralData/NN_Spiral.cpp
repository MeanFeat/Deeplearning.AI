#include "win32_ExpertSystems.h"
#include <time.h>

#define WINWIDTH 200
#define WINHEIGHT 200
#define WINHALFWIDTH WINWIDTH * 0.5f
#define WINHALFHEIGHT WINHEIGHT * 0.5f
#define SCALE 50

global_variable bool globalRunning = true;
global_variable bool discreteOutput = false;
global_variable bool plotData = true;
global_variable int winTitleHeight = 10;
static time_t startTime;
static time_t currentTime;

Buffer backBuffer;
Net neural;
NetTrainer trainer;

global_variable float GraphZoom = 1.f;

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
		case 'D':
			discreteOutput = !discreteOutput;
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
		case 'P':
			plotData = !plotData;
			break;
		default:
			break;
		}
	} break;
	case WM_MBUTTONDOWN:
	{
		GraphZoom = 1.f;
	}
	case WM_MOUSEWHEEL:
	{
		if(((short)HIWORD(WParam)) / 120 > 0) {
			PostMessage(Window, WM_VSCROLL, SB_LINEUP, (LPARAM)0);
			GraphZoom += 0.5;
		}
		if(((short)HIWORD(WParam)) / 120 < 0) {
			PostMessage(Window, WM_VSCROLL, SB_LINEDOWN, (LPARAM)0);
			GraphZoom *= 0.5;
		}
	}
	break;
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

void PlotData(MatrixXf X, MatrixXf Y) {
	for(int i = 0; i < X.cols(); ++i) {
		DrawFilledCircle(backBuffer, int(WINHALFWIDTH + X(0, i) * SCALE), int(WINHALFHEIGHT + -X(1, i) * SCALE), SCALE * 0.11f, Color(255, 255, 255, 255));
		DrawFilledCircle(backBuffer, int(WINHALFWIDTH + X(0, i) * SCALE),
						 int(WINHALFHEIGHT + -X(1, i) * SCALE), SCALE * 0.08f,
						 (Y(0, i) > 0.f ? positiveColor : negativeColor) - Color(50, 50, 50, 50));
	}
}

void DrawOutputToScreen(MatrixXf screenCoords) {
	MatrixXf h = neural.ForwardPropagation(screenCoords);
	int *pixel = (int *)backBuffer.memory;
	for(int i = 0; i < h.cols(); ++i) {
		float percent = (*(h.data() + i));
		Color blended = Color(0, 0, 0, 0);
		switch(neural.GetParams().layerActivations.back()) {
		case Sigmoid:
			percent = (percent - 0.5f) * 2;
			break;
		case Tanh:
			break;
		case ReLU:
			break;
		default:
			break;
		}
		if(discreteOutput) {
			blended = percent < 0.f ? negativeColor : positiveColor;
		} else {
			blended = percent < 0.f ? Color(255, 255, 255, 255).Blend(negativeColor, -percent)
				: Color(255, 255, 255, 255).Blend(positiveColor, percent);
		}
		*pixel++ = blended.ToBit();
	}
}

void UpdateHistory(vector<float> &history) {
	history.push_back(min((trainer.GetCache().cost) * (WINHEIGHT-backBuffer.titleOffset), WINHEIGHT));
	if(history.size() >= WINWIDTH + WINWIDTH) {
		for(int i = 1; i < (int)history.size(); i += 2) {
			history.erase(history.begin() + i);
		}
	}
}

MatrixXf BuildPolynomials(MatrixXf m) {
	MatrixXf temp = MatrixXf(m.rows() + 3, m.cols());
	temp << m, MatrixXf(m.array().pow(2)), MatrixXf(m.array().pow(2)).colwise().sum();
	return temp;
}

void UpdateDisplay(MatrixXf screenCoords, MatrixXf X, MatrixXf Y, vector<float> &history) {
	if(globalRunning) {
		DrawOutputToScreen(screenCoords);
		if(plotData) {
			PlotData(X, Y);
		}
		vector<float> zoomedHist;
		for(int i = GraphZoom > 1.f ? int(GraphZoom * 50) : 0; i < (int)history.size() - 1; ++i) {
			zoomedHist.push_back(min(WINHEIGHT, history[i] * GraphZoom));
		}
		DrawHistory(backBuffer, zoomedHist, Color(200,100,100,255));
	}
}

void UpdateWinTitle(int &steps, HWND window) {
	time(&currentTime);
	char s[255];
	sprintf_s(s, "SpiralData || Epoch %d | Time: %0.1f | Cost %0.10f | LearnRate %0.3f | RegTerm %0.3f | "
			  , steps++, difftime(currentTime, startTime), trainer.GetCache().cost, trainer.GetTrainParams().learningRate, trainer.GetTrainParams().regTerm);
	
	/*for(int l = 0; l < (int)neural.GetParams().layerSizes.size(); ++l) {
		char layer[255];
		sprintf_s(layer, "[%d]", neural.GetParams().layerSizes[l]);
		strcat_s(s, layer);
	}*/
	SetWindowText(window, LPCSTR(s));
}

int CALLBACK WinMain(HINSTANCE Instance, HINSTANCE PrevInstance, LPSTR CommandLine, int ShowCode) {
	MatrixXf X;//= (MatrixXf)BuildMatFromFile("Spiral.csv"); write_binary("Spiral_64.dat", X);
	MatrixXf Y;//= (MatrixXf)BuildMatFromFile("SpiralLabels.csv"); write_binary("SpiralLabels_64.dat", Y);

#if x64
	read_binary("Spiral_64.dat", X);
	read_binary("SpiralLabels_64.dat", Y);
#else
	read_binary("Spiral.dat", X);
	read_binary("SpiralLabels.dat", Y);
#endif

	X = BuildPolynomials(X);

	//X.conservativeResize(int(X.rows()), int(X.cols()*0.25));
	//Y.conservativeResize(int(Y.rows()), int(Y.cols()*0.25));

	WNDCLASSA winClass = {};
	InitializeWindow(&winClass, Instance, Win32MainWindowCallback, &backBuffer, WINWIDTH, WINHEIGHT, "NN_Spiral");
	MatrixXf screenCoords = BuildPolynomials(BuildDisplayCoords(backBuffer, SCALE).transpose());


	if(RegisterClassA(&winClass)) {
		HWND window = CreateWindowExA(0, winClass.lpszClassName, "NNet||",
									  WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MAXIMIZEBOX | WS_THICKFRAME | WS_VISIBLE, CW_USEDEFAULT, CW_USEDEFAULT,
									  WINWIDTH*4, WINHEIGHT*4, 0, 0, Instance, 0);

		neural = Net((int)X.rows(), { 8,8 }, (int)Y.rows(), {
			Tanh,
			Tanh,
			Tanh });
		initParallel();
		setNbThreads(4);
		trainer = NetTrainer(&neural, &X, &Y, 0.15f,
							 1.25f, 
							 20.f);

		time(&startTime);
		HDC deviceContext = GetDC(window);
		vector<float> history;
		int steps = 0;

		//Main Loop
		while(globalRunning) {
			for(int epoch = 0; epoch < 1; ++epoch) {
				Win32ProcessPendingMessages();
				if(!globalRunning) {
					break;
				}
				trainer.UpdateSingleStep();
				UpdateHistory(history);
				UpdateWinTitle(steps, window);
			}

			UpdateDisplay(screenCoords, X, Y, history);
			Win32DisplayBufferInWindow(deviceContext, window, backBuffer);
		}
		DeleteDC(deviceContext);
	}
	return EXIT_SUCCESS;
}
