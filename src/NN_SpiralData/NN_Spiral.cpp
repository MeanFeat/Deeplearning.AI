#include "win32_ExpertSystems.h"
#include "stdMat.h"
#define WINWIDTH 200
#define WINHEIGHT 200
#define WINHALFWIDTH WINWIDTH * 0.5f
#define WINHALFHEIGHT WINHEIGHT * 0.5f
#define SCALE 50

global_variable bool globalRunning = true;
global_variable bool discreteOutput = false;
global_variable bool plotData = true;
void *backBuffer;
Net neural;
global_variable float GraphZoom = 1.f;
BITMAPINFO bitmapInfo = { 0 };
global_variable Color positiveColor = Color(100, 167, 211, 255);
global_variable Color negativeColor = Color(255, 184, 113, 255);

internal void Win32DisplayBufferInWindow(void *Buffer, HDC DeviceContext, HWND hwind) {
	RECT winRect = {};
	GetWindowRect(hwind, &winRect);
	StretchDIBits(DeviceContext, 0, 0, winRect.right - winRect.left, winRect.bottom - winRect.top, 0, 0, WINWIDTH, WINHEIGHT, Buffer, &bitmapInfo, DIB_RGB_COLORS, SRCCOPY);
}

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
			neural.ModifyLearningRate(0.02f);
			break;
		case 'S':
			neural.ModifyLearningRate(-0.02f);
			break;
		case 'Q':
			neural.ModifyRegTerm(0.02f);
			break;
		case 'A':
			neural.ModifyRegTerm(-0.02f);
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

internal void Win32ProcessPendingMessages() {
	MSG Message;
	while(PeekMessage(&Message, 0, 0, 0, PM_REMOVE)) {
		TranslateMessage(&Message);
		DispatchMessageA(&Message);
	}
}

void PlotData(MatrixXf X, MatrixXf Y) {
	for(int i = 0; i < X.cols(); ++i) {
		DrawFilledCircle(backBuffer, WINWIDTH, int(WINHALFWIDTH + X(0, i) * SCALE), int(WINHALFHEIGHT + -X(1, i) * SCALE), SCALE * 0.11f, Color(255, 255, 255, 255));
		DrawFilledCircle(backBuffer, WINWIDTH, int(WINHALFWIDTH + X(0, i) * SCALE),
						 int(WINHALFHEIGHT + -X(1, i) * SCALE), SCALE * 0.08f,
						 (Y(0, i) > 0.f ? positiveColor : negativeColor) - Color(50, 50, 50, 50));
	}
}

MatrixXf BuildDisplayCoords() {
	MatrixXf out(WINWIDTH * WINHEIGHT, 2);
	VectorXf row(WINWIDTH);
	VectorXf cols(WINWIDTH * WINHEIGHT);
	for(int x = 0; x < WINWIDTH; ++x) {
		row(x) = float((x - WINHALFWIDTH) / SCALE);
	}
	for(int y = 0; y < WINHEIGHT; ++y) {
		for(int x = 0; x < WINWIDTH; ++x) {
			cols(y*WINWIDTH + x) = float((y - WINHALFWIDTH) / SCALE);
		}
	}
	out << row.replicate(WINHEIGHT, 1), cols;
	out.col(1) *= -1.f;
	return out;
}

void InitializeWindow(WNDCLASSA *winclass, HINSTANCE instance) {
	bitmapInfo.bmiHeader.biSize = sizeof(bitmapInfo.bmiHeader);
	bitmapInfo.bmiHeader.biHeight = WINHEIGHT;
	bitmapInfo.bmiHeader.biWidth = WINWIDTH;
	bitmapInfo.bmiHeader.biPlanes = 1;
	bitmapInfo.bmiHeader.biBitCount = 32;
	bitmapInfo.bmiHeader.biCompression = BI_RGB;
	backBuffer = malloc(bitmapInfo.bmiHeader.biHeight * bitmapInfo.bmiHeader.biWidth * 4);
	winclass->style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
	winclass->lpfnWndProc = Win32MainWindowCallback;
	winclass->hInstance = instance;
	winclass->lpszClassName = "HPlanarClassificationClass";
}

void DrawOutputToScreen(MatrixXf screenCoords) {
	MatrixXf h = neural.ForwardPropagation(screenCoords, false);
	int *pixel = (int *)backBuffer;
	for(int i = 0; i < h.cols(); ++i) {
		float percent = (*(h.data() + i));
		Color blended = Color(0, 0, 0, 0);
		switch(neural.GetParams().layerActivations.back()) {
		case Sigmoid:
			percent = (percent - 0.5f) * 2;
			break;
		case Tanh:
			break;
		default:
			break;
		}
		if(discreteOutput) {
			blended = percent < 0.f ? negativeColor : positiveColor;
		} else {
			blended = percent < 0.f ? Color(255, 255, 255, 255).Blend(negativeColor, tanh(-percent * 2))
				: Color(255, 255, 255, 255).Blend(positiveColor, tanh(percent * 2));
		}
		*pixel++ = blended.ToBit();
	}
}

void ClearScreen(MatrixXf screenCoords) {
	int *pixel = (int *)backBuffer;
	for(int i = 0; i < WINHEIGHT * WINWIDTH; ++i) {
		*pixel++ = Color(0, 0, 0, 0).ToBit();
	}
}

void UpdateHistory(vector<float> &history) {
	history.push_back(min((neural.GetCache().cost) * WINHEIGHT, WINHEIGHT));
	if(history.size() >= WINWIDTH*2) {
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
		DrawHistory(backBuffer, WINWIDTH, zoomedHist);
	}
}

void UpdateWinTitle(int &steps, HWND window) {
	char s[255];
	sprintf_s(s, "SpiralData || Epoch %d | Cost %0.10f | LearnRate %0.3f | RegTerm %0.3f | LayerSizes: "
			  , steps++, neural.GetCache().cost, neural.GetParams().learningRate, neural.GetParams().regTerm);
	
	for(int l = 0; l < (int)neural.GetParams().layerSizes.size(); ++l) {
		char layer[255];
		sprintf_s(layer, "[%d]", neural.GetParams().layerSizes[l]);
		strcat_s(s, layer);
	}
	SetWindowText(window, LPCSTR(s));
}

int CALLBACK WinMain(HINSTANCE Instance, HINSTANCE PrevInstance, LPSTR CommandLine, int ShowCode) {
	MatrixXf X;// = (MatrixXf)BuildMatFromFile("Spiral.txt"); X.transposeInPlace(); write_binary("Spiral.dat", X);
	MatrixXf Y;// = (MatrixXf)BuildMatFromFile("SpiralLabels.txt"); Y.transposeInPlace(); write_binary("SpiralLabels.dat", Y);
	read_binary("Spiral.dat", X);
	read_binary("SpiralLabels.dat", Y);
	WNDCLASSA winClass = {};
	InitializeWindow(&winClass, Instance);
	MatrixXf screenCoords = BuildDisplayCoords().transpose();

	//X.conservativeResize(int(X.rows()), int(X.cols()*0.25));
	//Y.conservativeResize(int(Y.rows()), int(Y.cols()*0.25));

	X = BuildPolynomials(X);
	screenCoords = BuildPolynomials(screenCoords);

	if(RegisterClassA(&winClass)) {
		HWND window = CreateWindowExA(0, winClass.lpszClassName, "NNet||",
									  WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MAXIMIZEBOX | WS_THICKFRAME | WS_VISIBLE, CW_USEDEFAULT, CW_USEDEFAULT,
									  WINWIDTH*4, WINHEIGHT*4, 0, 0, Instance, 0);

		neural.InitializeParameters(X.rows(), { 8,8 }, Y.rows(), {
			Tanh,
			Tanh,
			Tanh },
			0.25f,
			0.8f);

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
				neural.UpdateSingleStep(X, Y);
				UpdateHistory(history);
				UpdateWinTitle(steps, window);
			}

			UpdateDisplay(screenCoords, X, Y, history);
			Win32DisplayBufferInWindow(backBuffer, deviceContext, window);
		}

		DeleteDC(deviceContext);
	}
	return EXIT_SUCCESS;
}
