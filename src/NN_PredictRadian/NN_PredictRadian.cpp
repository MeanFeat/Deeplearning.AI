#include "win32_ExpertSystems.h"
#include "stdMat.h"
#include "windowsx.h"

#define WINWIDTH 350
#define WINHEIGHT WINWIDTH
#define WINHALFWIDTH int((WINWIDTH-1)*0.5f)
#define WINHALFHEIGHT WINHALFWIDTH

global_variable bool globalRunning = true;
global_variable bool training = false;
global_variable bool drawOutput = true;
global_variable float prediction = 0.f;
float mouseX = WINHALFWIDTH;
float mouseY = WINHALFHEIGHT - 100;

global_variable Color positiveColor = Color(100, 167, 211, 255);
global_variable Color negativeColor = Color(255, 184, 113, 255);

void *backBuffer;
Net neural;
BITMAPINFO bitmapInfo = { 0 };

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
		case 'T':
			training = !training;
			break;
		case 'D':
			drawOutput = !drawOutput;
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
		default:
			break;
		}
	} break;

	case WM_MOUSEMOVE:
	{
		if(DWORD(WParam) & MK_LBUTTON) {
			mouseX = float(GET_X_LPARAM(LParam));
			mouseY = float(GET_Y_LPARAM(LParam));
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

MatrixXf BuildDisplayCoords() {
	MatrixXf out(WINWIDTH * WINHEIGHT, 2);
	VectorXf row(WINWIDTH);
	VectorXf cols(WINWIDTH * WINHEIGHT);
	for(int x = 0; x < WINWIDTH; ++x) {
		row(x) = float((x - WINHALFWIDTH));
	}
	for(int y = 0; y < WINHEIGHT; ++y) {
		for(int x = 0; x < WINWIDTH; ++x) {
			cols(y*WINWIDTH + x) = float((y - WINHALFWIDTH));
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
	winclass->lpszClassName = "HNN_PredictRadian";
}

void ClearScreen() {
	int *pixel = (int *)backBuffer;
	for(int i = 0; i < WINHEIGHT * WINWIDTH; i+=4) {
		*pixel++ = Color(0, 0, 0, 0).ToBit();
		*pixel++ = Color(0, 0, 0, 0).ToBit();
		*pixel++ = Color(0, 0, 0, 0).ToBit();
		*pixel++ = Color(0, 0, 0, 0).ToBit();
	}
}

MatrixXf NormalizeData(MatrixXf m) {
	auto a = MatrixXf((m.row(0).array().pow(2) + m.row(1).array().pow(2)));
	auto b = MatrixXf(a.array().sqrt());
	auto c = MatrixXf(2, b.cols());
	c << b, b;
	MatrixXf normal = MatrixXf(m.array() / c.array());
	
	return normal;
}

void DrawOutputToScreen(MatrixXf screenCoords) {
	MatrixXf h = neural.ForwardPropagation(NormalizeData(screenCoords), false);
	int *pixel = (int *)backBuffer;
	for(int i = 0; i < h.cols(); ++i) {
		float percent = *(h.data() + i);
		Color blended = percent < 0.f ? Color(255, 255, 255, 255).Blend(negativeColor, -percent)
			: Color(255, 255, 255, 255).Blend(positiveColor, percent);
		*pixel++ = blended.ToBit();
	}
}


void UpdateDisplay(MatrixXf screenCoords, MatrixXf X, MatrixXf Y, vector<float> &history) {
	if(globalRunning) {
		if(drawOutput) {
			DrawOutputToScreen(screenCoords);
		} else {
			ClearScreen();
		}
		DrawLine(backBuffer, WINWIDTH, WINHALFWIDTH, WINHALFHEIGHT, mouseX, float(WINHEIGHT - mouseY), Color(0, 0, 255, 255));
		int predX = int(sin(prediction*Pi32) * 100.f);
		int predY = int(cos(prediction*Pi32) * 100.f);
		DrawLine(backBuffer, WINWIDTH, WINHALFWIDTH, WINHALFHEIGHT, float(WINHALFWIDTH)+predX, float(WINHALFHEIGHT)+predY, Color(255, 0, 0, 0));
		
	}	
}

void UpdateWinTitle(int &steps, HWND window) {
	char s[255];
	sprintf_s(s, "Epoch %d|Cost %0.5f|LR %0.2f|RT %0.2f "
			  , steps++, neural.GetCache().cost, neural.GetParams().learningRate, neural.GetParams().regTerm);
	char r[255];	
	sprintf_s(r, " |%0.2f|%0.2f| ", atan2((mouseX - WINHALFWIDTH), -(mouseY - WINHALFHEIGHT)), prediction*Pi32);
	strcat_s(s, r);
	SetWindowText(window, LPCSTR(s));
}

MatrixXf BuildRaidans(MatrixXf screenCoords) {
	MatrixXf out = MatrixXf(1, screenCoords.cols());
	for (int i = 0; i < screenCoords.cols()-1;++i){
		out(0,i) = atan2((screenCoords(0, i)), -(screenCoords(1, i))) / Pi32;
	}
	return out;
}

void UpdatePrediction() {
	MatrixXf mouse = MatrixXf(2,1);
	mouse(0, 0) = mouseX-WINHALFWIDTH;
	mouse(1, 0) = mouseY-WINHALFHEIGHT;
	MatrixXf h = neural.ForwardPropagation(NormalizeData(mouse/WINHALFHEIGHT), false);
	prediction = h(0);
}

int CALLBACK WinMain(HINSTANCE Instance, HINSTANCE PrevInstance, LPSTR CommandLine, int ShowCode) {
	MatrixXf X;
	MatrixXf Y;
	WNDCLASSA winClass = {};
	InitializeWindow(&winClass, Instance);
	MatrixXf screenCoords = BuildDisplayCoords().transpose();	

	//write_binary("Radian.dat", MatrixXf(screenCoords.array() / WINHALFHEIGHT));
	//write_binary("RadianLabels.dat", BuildRaidans(MatrixXf(screenCoords / WINHALFHEIGHT)));

	read_binary("Radian.dat", X);	
	read_binary("RadianLabels.dat", Y);
	X = NormalizeData(X);

	if(RegisterClassA(&winClass)) {
		HWND window = CreateWindowExA(0, winClass.lpszClassName, "NNet||",
									  WS_OVERLAPPED | WS_SYSMENU |WS_VISIBLE, CW_USEDEFAULT, CW_USEDEFAULT,
									  WINWIDTH, WINHEIGHT, 0, 0, Instance, 0);
		
		neural.InitializeParameters(X.rows(), { 10,5 }, Y.rows(), {
			Tanh,Tanh,
			Tanh },
			0.05f,
			0.0f);

		HDC deviceContext = GetDC(window);
		vector<float> history;
		int steps = 0;
		//Main Loop
		while(globalRunning) {
			int epochCount = 1;
			for(int epoch = 0; epoch < epochCount; ++epoch) {
				Win32ProcessPendingMessages();
				if(!globalRunning) {
					break;
				}
				neural.UpdateSingleStep(X, Y);
				UpdatePrediction();
				UpdateWinTitle(steps, window);
				epochCount = training ? 1000 : 1;
			}
			UpdateDisplay(screenCoords, X, Y, history);
			Win32DisplayBufferInWindow(backBuffer, deviceContext, window);
		}
		DeleteDC(deviceContext);
	}
	return EXIT_SUCCESS;
}
