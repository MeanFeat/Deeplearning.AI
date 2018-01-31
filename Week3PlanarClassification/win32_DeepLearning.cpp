#include "win32_DeepLearning.h"
#define WINWIDTH 500
#define WINHEIGHT 500
#define WINHALFWIDTH WINWIDTH * 0.5f
#define WINHALFHEIGHT WINHEIGHT * 0.5f
#define SCALE 50

global_variable bool globalRunning = true;
void *backBuffer;
BITMAPINFO bitmapInfo = {0};


internal void Win32DisplayBufferInWindow(void *Buffer, HDC DeviceContext) {
	StretchDIBits(DeviceContext, 0, 0, WINWIDTH, WINWIDTH, 0, 0, WINWIDTH, WINHEIGHT, Buffer, &bitmapInfo, DIB_RGB_COLORS, SRCCOPY);
}

internal LRESULT CALLBACK Win32MainWindowCallback(HWND Window, UINT Message, WPARAM WParam, LPARAM LParam) {
	LRESULT Result = 0;
	switch(Message) {
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
		switch(Message.message) {
			case WM_QUIT:
			{
				globalRunning = false;
			} break;
			default:
			{
				TranslateMessage(&Message);
				DispatchMessageA(&Message);
			} break;
		}
	}
}

void PlotData(MatrixXf X, MatrixXf Y) {
	for(int i = 0; i < X.cols(); i++) {
		Assert(X(1, i) != X(0, i));
		DrawFilledCircle(backBuffer, WINWIDTH, int((WINWIDTH / 2) + X(0, i) * SCALE), int((WINHEIGHT / 2) + -X(1, i) * SCALE), SCALE * 0.275f, Color(0, 0, 0, 255));
		DrawFilledCircle(backBuffer, WINWIDTH, int((WINWIDTH / 2) + X(0, i) * SCALE), int((WINHEIGHT / 2) + -X(1, i) * SCALE), SCALE * 0.2f, Y(i) == 1 ? Color(0, 255, 255, 255) : Color(255, 0, 0, 255));
	}
}

MatrixXf BuildDisplayCoords() {
	MatrixXf out(WINWIDTH * WINHEIGHT, 2);
	VectorXf row(WINWIDTH);
	VectorXf cols(WINWIDTH * WINHEIGHT);
	for(int x = 0; x < WINWIDTH; x++) {
		row(x) = float(x - WINHALFWIDTH);
	}
	for(int y = 0; y < WINHEIGHT; y++) {
		for(int x = 0; x < WINWIDTH; x++) {
			cols(y*WINWIDTH + x) = float((y - WINHALFWIDTH));;
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
	MatrixXf temp = BuildDisplayCoords();
	if(RegisterClassA(&winClass)) {
		HWND window = CreateWindowExA(0, winClass.lpszClassName, "PlanarClassification",
									  WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX | WS_MAXIMIZEBOX | WS_VISIBLE, CW_USEDEFAULT, CW_USEDEFAULT,
									  WINWIDTH, WINHEIGHT, 0, 0, Instance, 0);
		
		while(globalRunning) {
			Win32ProcessPendingMessages();
			PlotData(X, Y);
			HDC deviceContext = GetDC(window);
			Win32DisplayBufferInWindow(backBuffer, deviceContext);
			DeleteDC(deviceContext);
		}
	}
	return EXIT_SUCCESS;
}
