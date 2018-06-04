#include "win32_DeepLearning.h"
#include "stdMat.h"
#define WINWIDTH 800
#define WINHEIGHT 800
#define WINHALFWIDTH WINWIDTH * 0.5f
#define WINHALFHEIGHT WINHEIGHT * 0.5f
#define SCALE 80

global_variable bool globalRunning = true;
void *backBuffer;
BITMAPINFO bitmapInfo = { 0 };
global_variable Color positiveColor = Color(0, 0, 255, 255);
global_variable Color negativeColor = Color(255, 0, 0, 255);

internal void Win32DisplayBufferInWindow(void *Buffer, HDC DeviceContext) {
	StretchDIBits(DeviceContext, 0, 0, WINWIDTH, WINWIDTH, 0, 0, WINWIDTH, WINHEIGHT, Buffer, &bitmapInfo, DIB_RGB_COLORS, SRCCOPY);
}

internal LRESULT CALLBACK Win32MainWindowCallback(HWND Window, UINT Message, WPARAM WParam, LPARAM LParam) {
	LRESULT Result = 0;
	switch(Message) {
	case WM_DESTROY:
	case WM_CLOSE:
	{
		globalRunning = false;
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
		DrawFilledCircle(backBuffer, WINWIDTH, int((WINWIDTH / 2) + X(0, i) * SCALE), int((WINHEIGHT / 2) + -X(1, i) * SCALE), SCALE * 0.15f, Color(0, 0, 0, 255));
		DrawFilledCircle(backBuffer, WINWIDTH, int((WINWIDTH / 2) + X(0, i) * SCALE), int((WINHEIGHT / 2) + -X(1, i) * SCALE), SCALE * 0.1f, Y(i) == 1 ? positiveColor : negativeColor);
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
			cols(y*WINWIDTH + x) = float((y - WINHALFWIDTH)/SCALE);;
		}
	}
	out << row.replicate(WINHEIGHT, 1), cols;
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

int CALLBACK WinMain(HINSTANCE Instance, HINSTANCE PrevInstance, LPSTR CommandLine, int ShowCode) {
	MatrixXf X;// = (MatrixXf)BuildMatFromFile("new.txt"); write_binary("planar.dat", X);
	MatrixXf Y;// = (MatrixXf)BuildMatFromFile("newL.txt"); write_binary("planarLabels.dat", Y);
	read_binary("planar.dat", X);
	read_binary("planarLabels.dat", Y);
	WNDCLASSA winClass = {};
	InitializeWindow(&winClass, Instance);
	MatrixXf screenCoords = BuildDisplayCoords().transpose();
	if(RegisterClassA(&winClass)) {
		HWND window = CreateWindowExA(0, winClass.lpszClassName, "PlanarClassification",
									  WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_VISIBLE, CW_USEDEFAULT, CW_USEDEFAULT,
									  WINWIDTH, WINHEIGHT, 0, 0, Instance, 0);
		Net neural;
		neural.InitializeParameters(X.rows(), { 9,9 }, Y.rows(), {
			Tanh,
			Tanh,
			Sigmoid },
			0.125f);

		while(globalRunning) {
			Win32ProcessPendingMessages();
			for(int epoch = 0; epoch < 10; epoch++) {
				neural.UpdateSingleStep(X, Y);
			}
			MatrixXf h = neural.ForwardPropagation(screenCoords, false);
			int *pixel = (int *)backBuffer;

			for(int i = 0; i < h.cols(); i++) {
				float percent = *(h.data() + i);
#if 1
				Color blended = positiveColor.Blend(negativeColor, percent);
#else
				Color blended = percent < 0.5f ? positiveColor : negativeColor;
#endif
				*pixel++ = blended.ToBit();
			}

			HDC deviceContext = GetDC(window);
			PlotData(X, Y);
			Win32DisplayBufferInWindow(backBuffer, deviceContext);
			DeleteDC(deviceContext);
		}
		
	}
	return EXIT_SUCCESS;
}
