#include "win32_ExpertSystems.h"
#include "stdMat.h"
#include "windowsx.h"

#define WINWIDTH 350
#define WINHEIGHT WINWIDTH
#define WINHALFWIDTH int((WINWIDTH-1)*0.5f)
#define WINHALFHEIGHT WINHALFWIDTH

global_variable bool globalRunning = true;
global_variable bool training = true;
global_variable bool drawOutput = true;
global_variable bool plotData = true;

global_variable vector<float> predictions;
float mouseX = WINHALFWIDTH;
float mouseY = WINHALFHEIGHT + 100;
int windowTitleHeight = -25;
global_variable Color positiveColor = Color(100, 167, 211, 255);
global_variable Color negativeColor = Color(255, 184, 113, 255);

void *backBuffer;
Net neural;
BITMAPINFO bitmapInfo = { 0 };

internal void Win32DisplayBufferInWindow(void *Buffer, HDC DeviceContext, HWND hwind) {
	RECT winRect = {};
	GetWindowRect(hwind, &winRect);
	StretchDIBits(DeviceContext, 0, windowTitleHeight, winRect.right - winRect.left, winRect.bottom - winRect.top, 0, 0, WINWIDTH, WINHEIGHT, Buffer, &bitmapInfo, DIB_RGB_COLORS, SRCCOPY);
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
		case 'P':
			plotData = !plotData;
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
	case WM_LBUTTONDOWN: //fall through
	case WM_MOUSEMOVE:
	{
		if(DWORD(WParam) & MK_LBUTTON) {
			mouseX = float(GET_X_LPARAM(LParam));
			mouseY = float(GET_Y_LPARAM(LParam)) - windowTitleHeight;
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
		*pixel++ = Color(10, 10, 10, 0).ToBit();
		*pixel++ = Color(10, 10, 10, 0).ToBit();
		*pixel++ = Color(10, 10, 10, 0).ToBit();
		*pixel++ = Color(10, 10, 10, 0).ToBit();
	}
}

MatrixXf NormalizeData(MatrixXf m) {
	MatrixXf a = (m.row(0).array().pow(2) + m.row(1).array().pow(2)).array().sqrt();
	MatrixXf b = MatrixXf(2, a.cols());
	b << a, a;
	return MatrixXf(m.array() / b.array());
}

Color GetColorBlend(float percent) {		
	return percent < 0.f ? Color(255, 255, 255, 255).Blend(negativeColor, -percent)
		: Color(255, 255, 255, 255).Blend(positiveColor, percent);;
}

void PlotData(MatrixXf X, MatrixXf Y) {
	for(int i = 0; i < X.cols(); ++i) {
		//DrawFilledCircle(backBuffer, WINWIDTH, int(WINHALFWIDTH + X(0, i)), int(WINHALFHEIGHT + -X(1, i)), 11.f, Color(0, 0, 0, 0));
		DrawFilledCircle(backBuffer, WINWIDTH, int(WINHALFWIDTH + X(0, i)), int(WINHALFHEIGHT + -X(1, i)), 10.f, GetColorBlend(Y(0, i)));
	}
}

void DrawOutputToScreen(MatrixXf normScreen) {
	MatrixXf h = neural.ForwardPropagation(normScreen, false);
	int *pixel = (int *)backBuffer;
	for(int i = 0; i < normScreen.cols(); ++i) {
		float percent = h(0,i);

		*pixel++ = GetColorBlend(percent).ToBit();
	}
}

void UpdateDisplay(MatrixXf screenCoords, MatrixXf X, MatrixXf Y, vector<float> &history) {
	if(globalRunning) {
		if(drawOutput) {
			DrawOutputToScreen(screenCoords);
		} else {
			ClearScreen();
		}
		if (plotData){
			PlotData(X*120, Y);
		}

		DrawLine(backBuffer, WINWIDTH, WINHALFWIDTH, WINHALFHEIGHT, mouseX, float(WINHEIGHT - mouseY), Color(0, 0, 255, 255));
		for (int i = 0; i < (int)predictions.size();++i){
			int predX = int(sin(predictions[i] * Pi32) * 100.f);
			int predY = int(cos(predictions[i] * Pi32) * 100.f);
			DrawLine(backBuffer, WINWIDTH, WINHALFWIDTH, WINHALFHEIGHT, float(WINHALFWIDTH) + predX, float(WINHALFHEIGHT) - predY, Color(100, 0, 0, 0));
		}
		DrawHistory(backBuffer, WINWIDTH, history);
	}	
}

void UpdateWinTitle(int &steps, HWND window) {
	char s[255];
	sprintf_s(s, "Epoch %d|Cost %0.15f|LR %0.2f|RT %0.2f "
			  , steps, neural.GetCache().cost, neural.GetParams().learningRate, neural.GetParams().regTerm);
	char r[255];	
	sprintf_s(r, " |%0.2f|%0.2f| ", atan2((mouseX - WINHALFWIDTH), (mouseY - WINHALFHEIGHT)), predictions[0]*Pi32);
	strcat_s(s, r);
	SetWindowText(window, LPCSTR(s));
}

MatrixXf BuildRadians(MatrixXf m) {
	float coefficient = 1.f / Pi32;
	MatrixXf out = MatrixXf(1, m.cols());
	for(int i = 0; i < m.cols(); ++i) {
		out(0, i) = atan2((m(0, i)), (m(1, i))) * coefficient;
		//out(1, i) = -atan2((m(1, i)), -(m(0, i))) * coefficient;
		//out(2, i) = -atan2(-(m(1, i)), (m(0, i))) * coefficient;
	}
	return out;
}

void UpdatePrediction() {
	MatrixXf mouse = MatrixXf(2,1);
	mouse(0, 0) = mouseX-WINHALFWIDTH;
	mouse(1, 0) = mouseY-WINHALFHEIGHT;
	MatrixXf h = neural.ForwardPropagation(NormalizeData(mouse / WINHALFHEIGHT), false);
	predictions.clear();
	for (int i = 0; i < h.rows(); ++i){
		predictions.push_back(h(i, 0));
	}
}

#define clamp(x,lo,hi) min( hi, max(lo,x) )

MatrixXf CreateSparseData(int pointCount) {
	float piAdjusted = Pi32 - 0.01f;
	float delta = (2 * Pi32) / float(pointCount);
	MatrixXf out = MatrixXf(2, pointCount+1);
	for(int i = 0; i <= pointCount; i++) {
		float r = clamp(-Pi32 + (delta * i), -piAdjusted, piAdjusted);
		out(0,i) = sin(r);
		out(1,i) = cos(r);
	}
	return out;
}

void UpdateHistory(vector<float> &history) {
	history.push_back(min((neural.GetCache().cost) * WINHEIGHT, WINHEIGHT));
	if(history.size() >= WINWIDTH + WINWIDTH) {
		for(int i = 1; i < (int)history.size(); i += 2) {
			history.erase(history.begin() + i);
		}
	}
}


int CALLBACK WinMain(HINSTANCE Instance, HINSTANCE PrevInstance, LPSTR CommandLine, int ShowCode) {
	MatrixXf X;
	MatrixXf Y;
	WNDCLASSA winClass = {};
	InitializeWindow(&winClass, Instance);
	MatrixXf screenCoords = NormalizeData(BuildDisplayCoords().transpose());

	//write_binary("Radian.dat", MatrixXf(screenCoords.array() / WINHALFHEIGHT));
	//write_binary("RadianLabels.dat", BuildRaidans(MatrixXf(screenCoords / WINHALFHEIGHT)));
	//read_binary("Radian.dat", X);
	//read_binary("RadianLabels.dat", Y);
	//X = NormalizeData(X);

	X = NormalizeData(CreateSparseData(30));
	Y = BuildRadians(X);

	if(RegisterClassA(&winClass)) {
		HWND window = CreateWindowExA(0, winClass.lpszClassName, "NNet||",
									  WS_OVERLAPPED | WS_SYSMENU |WS_VISIBLE, CW_USEDEFAULT, CW_USEDEFAULT,
									  WINWIDTH, WINHEIGHT, 0, 0, Instance, 0);
		
		neural.InitializeParameters(X.rows(), { 15,10 }, Y.rows(), {
			Tanh, Tanh,
			Tanh },
			0.15f,
			0.2f);

		HDC deviceContext = GetDC(window);
		vector<float> history;
		int steps = 0;
		//Main Loop
		while(globalRunning) {
			Win32ProcessPendingMessages();	
			if (training) {
				neural.UpdateSingleStep(X, Y);
				steps++; 
				UpdateHistory(history);
			} else {
			}
			UpdateDisplay(screenCoords, X, Y, history);
			UpdatePrediction();
			UpdateWinTitle(steps, window);			
			Win32DisplayBufferInWindow(backBuffer, deviceContext, window);
		}
		DeleteDC(deviceContext);
	}
	return EXIT_SUCCESS;
}
