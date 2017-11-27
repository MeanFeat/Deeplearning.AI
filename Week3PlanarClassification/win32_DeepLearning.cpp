#include "win32_DeepLearning.h"

#define internal static 
#define local_persist static 
#define global_variable static

global_variable bool globalRunning = true;
static win32_offscreen_buffer GlobalBackbuffer;

internal win32_window_dimension Win32GetWindowDimension(HWND Window) {
	win32_window_dimension Result;
	RECT ClientRect;
	GetClientRect(Window, &ClientRect);
	Result.Width = ClientRect.right - ClientRect.left;
	Result.Height = ClientRect.bottom - ClientRect.top;
	return Result;
}

internal void Win32ResizeDIBSection(win32_offscreen_buffer *Buffer, int Width, int Height) {
	if(Buffer->Memory) {
		VirtualFree(Buffer->Memory, 0, MEM_RELEASE);
	}
	Buffer->Width = Width;
	Buffer->Height = Height;
	int BytesPerPixel = 4;
	Buffer->BytesPerPixel = BytesPerPixel;
	Buffer->Info.bmiHeader.biWidth = Buffer->Width;
	Buffer->Info.bmiHeader.biHeight = -Buffer->Height;
	Buffer->Info.bmiHeader.biPlanes = 1;
	Buffer->Info.bmiHeader.biBitCount = 32;
	Buffer->Info.bmiHeader.biCompression = BI_RGB;
	int BitmapMemorySize = (Buffer->Width*Buffer->Height)*BytesPerPixel;
	Buffer->Memory = VirtualAlloc(0, BitmapMemorySize, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
	Buffer->Pitch = Width*BytesPerPixel;
}

internal void Win32DisplayBufferInWindow(win32_offscreen_buffer *Buffer, HDC DeviceContext, int WindowWidth, int WindowHeight) {
	StretchDIBits(DeviceContext,
				  0, 0, WindowWidth, WindowHeight,
				  0, 0, Buffer->Width, Buffer->Height,
				  Buffer->Memory,
				  &Buffer->Info,
				  DIB_RGB_COLORS, SRCCOPY);
}

internal LRESULT CALLBACK Win32MainWindowCallback(HWND Window, UINT Message, WPARAM WParam, LPARAM LParam) {
	LRESULT Result = 0;
	switch(Message) {
		case WM_ACTIVATEAPP:
		{
			OutputDebugStringA("WM_ACTIVATEAPP\n");
		} break;
		case WM_PAINT:
		{
			PAINTSTRUCT Paint;
			HDC DeviceContext = BeginPaint(Window, &Paint);
			win32_window_dimension Dimension = Win32GetWindowDimension(Window);
			Win32DisplayBufferInWindow(&GlobalBackbuffer, DeviceContext, Dimension.Width, Dimension.Height);
			EndPaint(Window, &Paint);

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

internal void RenderWeirdGradient(win32_offscreen_buffer *Buffer, int BlueOffset, int GreenOffset) {
	uint8 *Row = (uint8 *)Buffer->Memory;
	for(int Y = 0; Y < Buffer->Height; ++Y) {
		uint32 *Pixel = (uint32 *)Row;
		for(int X = 0; X < Buffer->Width; ++X) {
			uint8 Blue = (uint8)(X + BlueOffset);
			uint8 Green = (uint8)(Y + GreenOffset);
			*Pixel++ = ((Green << 8) | Blue);
		}
		Row += Buffer->Pitch;
	}
}


void PlotData(HDC *hdc, MatrixXf X, MatrixXf Y) {
	for(int i = 0; i < X.cols(); i++) {
		Assert(X(1, i) != X(0, i));
		DrawFilledCircle(hdc, int((WINWIDTH / 2) + X(0, i) * SCALE), int((WINHEIGHT / 2) + -X(1, i) * SCALE), SCALE * 0.275f, Color(0, 0, 0, 255));
		DrawFilledCircle(hdc, int((WINWIDTH / 2) + X(0, i) * SCALE), int((WINHEIGHT / 2) + -X(1, i) * SCALE), SCALE * 0.2f,
						 Y(i) == 1 ? Color(0, 255, 255, 255) : Color(255, 0, 0, 255));
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

int CALLBACK WinMain(HINSTANCE Instance, HINSTANCE PrevInstance, LPSTR CommandLine, int ShowCode) {
	MatrixXf X;// = (MatrixXf)BuildMatFromFile("new.txt"); write_binary("planar.dat", X);
	MatrixXf Y;// = (MatrixXf)BuildMatFromFile("newL.txt"); write_binary("planarLabels.dat", Y);
	read_binary("planar.dat", X);
	read_binary("planarLabels.dat", Y);
	WNDCLASSA winClass = {};
	Win32ResizeDIBSection(&GlobalBackbuffer, WINWIDTH, WINHEIGHT);
	winClass.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
	winClass.lpfnWndProc = Win32MainWindowCallback;
	winClass.hInstance = Instance;
	winClass.lpszClassName = "HPlanarClassificationClass";
	int randColor = 0;
	if(RegisterClassA(&winClass)) {
		HWND window = CreateWindowExA(0, winClass.lpszClassName, "PlanarClassification",
									  WS_OVERLAPPEDWINDOW | WS_VISIBLE, CW_USEDEFAULT,CW_USEDEFAULT,
									  WINWIDTH, WINHEIGHT, 0, 0, Instance, 0);
		HDC DeviceContext = GetDC(window);
		MatrixXf temp = BuildDisplayCoords();
		while(globalRunning) {

			Win32ProcessPendingMessages();
			HDC hDCMem = CreateCompatibleDC(DeviceContext); 
			HBITMAP Membitmap = CreateCompatibleBitmap(DeviceContext, WINWIDTH, WINHEIGHT);
			SelectObject(hDCMem, Membitmap);			int randColor = int(rand() % 255);
			for(int i = 0; i < temp.rows(); i++) {
				SetPixelV(hDCMem, int(temp(i, 0) + WINHALFHEIGHT), int((temp(i, 1)) + WINHALFWIDTH), RGB(randColor, 150, 255));
			}
			PlotData(&hDCMem, X, Y);
			BitBlt(DeviceContext, 0, 0, WINWIDTH, WINHEIGHT, hDCMem, 0, 0, SRCCOPY);
		}
	}
	return EXIT_SUCCESS;
}
