#include "win32_DeepLearning.h"
#include "stdMat.h"
#define WINWIDTH 400
#define WINHEIGHT 400
#define WINHALFWIDTH WINWIDTH * 0.5f
#define WINHALFHEIGHT WINHEIGHT * 0.5f
#define SCALE 100

global_variable bool globalRunning = true;
global_variable bool discreteOutput = false;
void *backBuffer;
Net neural;
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
		case 'S':
			neural.SaveNetwork();
			break;
		case 'L':
			neural.LoadNetwork();
			break;
		default:
			break;
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

internal void Win32ProcessPendingMessages() {
	MSG Message;
	while(PeekMessage(&Message, 0, 0, 0, PM_REMOVE)) {
		TranslateMessage(&Message);
		DispatchMessageA(&Message);
	}
}

void PlotData(MatrixXf X, MatrixXf Y) {
	for(int i = 0; i < X.cols(); i++) {
		Assert(X(1, i) != X(0, i));
		DrawFilledCircle(backBuffer, WINWIDTH, int((WINWIDTH / 2) + X(0, i) * SCALE), int((WINHEIGHT / 2) + -X(1, i) * SCALE), SCALE * 0.075f, Color(255, 255, 255, 255));
		DrawFilledCircle(backBuffer, WINWIDTH, int((WINWIDTH / 2) + X(0, i) * SCALE), 
						 int((WINHEIGHT / 2) + -X(1, i) * SCALE), SCALE * 0.05f,
						 (Y(0,i) > 0.f ? positiveColor : negativeColor) - Color(50, 50,50,50));
	}
}

MatrixXf BuildDisplayCoords() {
	MatrixXf out(WINWIDTH * WINHEIGHT, 2);
	VectorXf row(WINWIDTH);
	VectorXf cols(WINWIDTH * WINHEIGHT);
	for(int x = 0; x < WINWIDTH; x++) {
		row(x) = float((x - WINHALFWIDTH) / SCALE);
	}
	for(int y = 0; y < WINHEIGHT; y++) {
		for(int x = 0; x < WINWIDTH; x++) {
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

void DrawOutputToScreen(MatrixXf screenCoords){

	MatrixXf h = neural.ForwardPropagation(screenCoords, false);
	int *pixel = (int *)backBuffer;
	for(int i = 0; i < h.cols(); i++) {
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
			blended = percent < 0.f ? Color(255, 255, 255, 255).Blend(negativeColor, tanh(-percent * 5))
				: Color(255, 255, 255, 255).Blend(positiveColor, tanh(percent * 5));
		}
		*pixel++ = blended.ToBit();
	}
}

void UpdateHistory(vector<float> &history) {
	history.push_back(neural.GetCache().cost * WINHEIGHT);
}

int CALLBACK WinMain(HINSTANCE Instance, HINSTANCE PrevInstance, LPSTR CommandLine, int ShowCode) {
	MatrixXf X;// = (MatrixXf)BuildMatFromFile("Spiral.txt"); X.transposeInPlace(); write_binary("Spiral.dat", X);
	MatrixXf Y;// = (MatrixXf)BuildMatFromFile("SpiralLabels.txt"); Y.transposeInPlace(); write_binary("SpiralLabels.dat", Y);
	read_binary("Spiral.dat", X);
	read_binary("SpiralLabels.dat", Y);

	WNDCLASSA winClass = {};
	InitializeWindow(&winClass, Instance);
	
	MatrixXf screenCoords = BuildDisplayCoords().transpose();
	if(RegisterClassA(&winClass)) {
		HWND window = CreateWindowExA(0, winClass.lpszClassName, "PlanarClassification",
									  WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_THICKFRAME | WS_VISIBLE, CW_USEDEFAULT, CW_USEDEFAULT,
									  WINWIDTH, WINHEIGHT, 0, 0, Instance, 0);
		
		neural.InitializeParameters(X.rows(), { 19,19 }, Y.rows(), {
			Tanh,
			Tanh,
			Tanh },
			0.125f);

		HDC deviceContext = GetDC(window);		
		vector<float> history;

		//Main Loop
		while(globalRunning) {
			Win32ProcessPendingMessages();
			for(int epoch = 0; epoch < 100; epoch++) {
				neural.UpdateSingleStep(X, Y);
				UpdateHistory(history);
			}
			DrawOutputToScreen(screenCoords);
			PlotData(X, Y);
			DrawHistory(backBuffer, WINWIDTH, history);
			Win32DisplayBufferInWindow(backBuffer, deviceContext, window);
		}

		DeleteDC(deviceContext);
	}
	return EXIT_SUCCESS;
}
