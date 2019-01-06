#include "win32_ExpertSystems.h"
#include <time.h>

#define TEST_CPU 0

#define WINWIDTH 200
#define WINHEIGHT 200
#define WINHALFWIDTH WINWIDTH * 0.5
#define WINHALFHEIGHT WINHEIGHT * 0.5
#define SCALE 50

global_variable bool globalRunning = true;
global_variable bool discreteOutput = false;
global_variable bool plotData = false;
global_variable int winTitleHeight = 10;
static time_t startTime;
static time_t currentTime;

Buffer backBuffer;
Net neural;

NetTrainer h_trainer;
d_NetTrainer d_trainer;

global_variable double GraphZoom = 1.f;

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
			h_trainer.ModifyLearningRate(0.02);
			d_trainer.ModifyLearningRate(0.02);
			break;
		case 'S':
			h_trainer.ModifyLearningRate(-0.02);
			d_trainer.ModifyLearningRate(-0.02);
			break;
		case 'Q':
			h_trainer.ModifyRegTerm(0.02);
			d_trainer.ModifyRegTerm(0.02);
			break;
		case 'A':
			h_trainer.ModifyRegTerm(-0.02);
			d_trainer.ModifyRegTerm(-0.02);
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

void PlotData(MatrixXd X, MatrixXd Y) {
	for(int i = 0; i < X.cols(); ++i) {
		DrawFilledCircle(backBuffer, int(WINHALFWIDTH + X(0, i) * SCALE), int(WINHALFHEIGHT + -X(1, i) * SCALE), SCALE * 0.11, Color(255, 255, 255, 255));
		DrawFilledCircle(backBuffer, int(WINHALFWIDTH + X(0, i) * SCALE),
						 int(WINHALFHEIGHT + -X(1, i) * SCALE), SCALE * 0.08,
						 (Y(0, i) > 0.f ? positiveColor : negativeColor) - Color(50, 50, 50, 50));
	}
}

void UpdateHistory(vector<double> &history, double cost) {
	history.push_back(min((cost) * (WINHEIGHT-backBuffer.titleOffset), WINHEIGHT));
	if(history.size() >= WINWIDTH + WINWIDTH) {
		for(int i = 1; i < (int)history.size(); i += 2) {
			history.erase(history.begin() + i);
		}
	}
}

MatrixXd BuildPolynomials(MatrixXd m) {
	MatrixXd temp = MatrixXd(m.rows() + 3, m.cols());
	temp << m, MatrixXd(m.array().pow(2)), MatrixXd(m.array().pow(2)).colwise().sum();
	return temp;
}

void DrawOutputToScreen(MatrixXd screenCoords) {
	MatrixXd h = neural.ForwardPropagation(screenCoords);
	int *pixel = (int *)backBuffer.memory;
	for(int i = 0; i < h.cols(); ++i) {
		double percent = (*(h.data() + i));
		Color blended = Color(0, 0, 0, 0);
		switch(neural.GetParams().layerActivations.back()) {
		case Sigmoid:
			percent = (percent - 0.5) * 2;
			break;
		case Tanh:
			break;
		case ReLU:
			break;
		default:
			break;
		}
		if(discreteOutput) {
			blended = percent < 0.0 ? negativeColor : positiveColor;
		} else {
			blended = percent < 0.0 ? Color(255, 255, 255, 255).Blend(negativeColor, -percent)
				: Color(255, 255, 255, 255).Blend(positiveColor, percent);
		}
		*pixel++ = blended.ToBit();
	}
}

void UpdateDisplay(MatrixXd screenCoords, MatrixXd X, MatrixXd Y, vector<double> &h_history, vector<double> &d_history, cudaStream_t *stream) {
	if(globalRunning) {
		//DrawOutputToScreen(screenCoords);
		d_trainer.Visualization((int *)backBuffer.memory, backBuffer.width, backBuffer.height, discreteOutput, stream);
		/*int *pixel = (int *)backBuffer.memory;
		for(int i = 0; i < WINWIDTH*WINHEIGHT; ++i) {
			*pixel++ = ((0 << 16) | ((0 << 8) | 0));
		}*/
		if(plotData) {
			PlotData(X, Y);
		}
		DrawHistory(backBuffer, h_history, Color(200, 100, 100, 255));
		DrawHistory(backBuffer, d_history, Color(100, 100, 200, 255));
	}
}

void UpdateWinTitle(int &steps, HWND window) {
	time(&currentTime);
	char s[255];
	sprintf_s(s, "SpiralData || Epoch %d | Time: %0.1f | h_Cost %0.10f | d_Cost %0.10f  "
			  , steps++, difftime(currentTime, startTime), h_trainer.GetCache().cost, d_trainer.GetCache().cost);
	SetWindowText(window, LPCSTR(s));
}

int CALLBACK WinMain(HINSTANCE Instance, HINSTANCE PrevInstance, LPSTR CommandLine, int ShowCode) {
	MatrixXd X;// = (MatrixXd)BuildMatFromFile("Spiral.csv"); write_binary("Spiral_64.dat", X);
	MatrixXd Y;// = (MatrixXd)BuildMatFromFile("SpiralLabels.csv"); write_binary("SpiralLabels_64.dat", Y);

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
	MatrixXd screenCoords = BuildPolynomials(BuildDisplayCoords(backBuffer, SCALE).transpose());


	if(RegisterClassA(&winClass)) {
		HWND window = CreateWindowExA(0, winClass.lpszClassName, "NNet||",
									  WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MAXIMIZEBOX | WS_THICKFRAME | WS_VISIBLE, CW_USEDEFAULT, CW_USEDEFAULT,
									  WINWIDTH*4, WINHEIGHT*4, 0, 0, Instance, 0);

		neural = Net((int)X.rows(), { 8,8 }, (int)Y.rows(), {
			Tanh,
			Tanh,
			Tanh });

		h_trainer = NetTrainer(&neural, &X, &Y, 0.25, 2.00, 20.0);
		d_trainer = d_NetTrainer(&neural, &X, &Y, 1.0, 2.0, 20.0);

		time(&startTime);
		HDC deviceContext = GetDC(window);
		vector<double> h_history;
		vector<double> d_history;
		int steps = 0;
		cudaStream_t stream;
		cudaStreamCreate(&stream);
		d_trainer.BuildVisualization(screenCoords, (int *)backBuffer.memory, backBuffer.width, backBuffer.height);
		//Main Loop
		while(globalRunning) {
			for(int epoch = 0; epoch < 100; ++epoch) {
				Win32ProcessPendingMessages();
				if(!globalRunning) {
					break;
				}
				h_trainer.UpdateSingleStep();
				d_trainer.UpdateSingleStep();
				UpdateHistory(h_history, h_trainer.GetCache().cost);
				UpdateHistory(d_history, d_trainer.GetCache().cost);
				UpdateWinTitle(steps, window);
			}
			UpdateDisplay(screenCoords, X, Y, h_history, d_history, &stream);
			Win32DisplayBufferInWindow(deviceContext, window, backBuffer);
		}
		DeleteDC(deviceContext);
	}
	return EXIT_SUCCESS;
}
