#include "win32_ExpertSystems.h"
#include "stdMat.h"
#include "windowsx.h"
#include <time.h>

#define WINWIDTH 44
#define WINHEIGHT 55
#define WINHALFWIDTH int((WINWIDTH)*0.5f)
#define WINHALFHEIGHT int((WINHEIGHT)*0.5f)

global_variable bool globalRunning = true;
global_variable bool isTraining = true;
global_variable bool drawOutput = false;
global_variable bool plotData = false;

static time_t startTime;
static time_t currentTime;

global_variable float rotX =0.f;
MatrixXf test = MatrixXf(1, 1);

Buffer backBuffer;
Net neural;
NetTrainer trainer;

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
		case 'Z':
			rotX += 0.01f;
			break;
		case 'X':
			rotX -= 0.01f;
			break;
		default:
			break;
		}
	} break;
	case WM_LBUTTONDOWN: //fall through
	case WM_MOUSEMOVE:
	{
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

	rotX = rotX > 1.f ? -1.f : rotX;
	rotX = rotX < -1.f ? 1.f : rotX;

	return Result;
}


void DrawMatrix(MatrixXf m) {
	int *pixel = (int *)backBuffer.memory;
	for(int i = 0; i < m.rows(); i += 3) {
		int r = int(m(i + 2, 0) * 255);
		int g = int(m(i + 1, 0) * 255);
		int b = int(m(i + 0, 0) * 255);
		*pixel++ = Color(r, g, b, 255).ToBit();
	}
}

MatrixXf BuildPolynomials(MatrixXf m) {
	MatrixXf temp0 = MatrixXf(m.rows() + 1, m.cols());
	temp0 << m, MatrixXf(m.array().pow(2));
	MatrixXf temp1 = MatrixXf(temp0.rows() + 2, temp0.cols());
	temp1 << temp0, temp0 * -1.f;
	return temp1;
}

void UpdateDisplay(MatrixXf X, MatrixXf Y, vector<float> &history, vector<float> &testHistory) {
	if(drawOutput) {		
		test(0, 0) = rotX;
		DrawMatrix(neural.ForwardPropagation(BuildPolynomials(test)));
	} else {
		ClearScreen(backBuffer);
	}
	if(plotData) {
	}
	DrawHistory(backBuffer, history, Color(200, 90, 90, 255));

}

void UpdateWinTitle(int &steps, HWND window) {
	time(&currentTime);
	char s[255];
	sprintf_s(s, "%d|T:%0.f|C:%0.10f|LR:%0.2f|RT:%0.2f|MX:%0.2f"
			  , steps, difftime(currentTime, startTime), trainer.GetCache().cost, trainer.GetTrainParams().learningRate, trainer.GetTrainParams().regTerm,rotX);

	SetWindowText(window, LPCSTR(s));
}


void UpdateHistory(vector<float> &history, float cost) {
	history.push_back(min((cost)* WINHEIGHT, WINHEIGHT));
	if(history.size() >= WINWIDTH + WINWIDTH) {
		for(int i = 1; i < (int)history.size(); i += 2) {
			history.erase(history.begin() + i);
		}
	}
}

MatrixXf readBMP(char* filename) {
	int i;
	FILE* f = fopen(filename, "rb");
	unsigned char info[54];
	fread(info, sizeof(unsigned char), 54, f); // read the 54-byte header
											   // extract image height and width from header
	int width = *(int*)&info[18];
	int height = *(int*)&info[22];

	int size = 3 * width * height;
	unsigned char* data = new unsigned char[size]; // allocate 3 bytes per pixel
	fread(data, sizeof(unsigned char), size, f); // read the rest of the data at once
	fclose(f);
	MatrixXf out = MatrixXf(size, 1);
	for(i = 0; i < size; i += 3) {
		out(i + 0, 0) = float(data[i + 0]);
		out(i + 1, 0) = float(data[i + 1]);
		out(i + 2, 0) = float(data[i + 2]);
	}
	return out;
}

int CALLBACK WinMain(HINSTANCE Instance, HINSTANCE PrevInstance, LPSTR CommandLine, int ShowCode) {
	
	WNDCLASSA winClass = {};
	InitializeWindow(&winClass, Instance, Win32MainWindowCallback, &backBuffer, WINWIDTH, WINHEIGHT, "NN_PredictRadian");
	MatrixXf X = MatrixXf(1, 9);
	MatrixXf Y = MatrixXf(WINWIDTH*WINHEIGHT * 3, 9);

	Y << readBMP("D:\\Gamedev\\ExpertSystems.AI\\src\\NN_DoomMarine\\doom.0.bmp"),
		readBMP("D:\\Gamedev\\ExpertSystems.AI\\src\\NN_DoomMarine\\doom.0.785398.bmp"),
		readBMP("D:\\Gamedev\\ExpertSystems.AI\\src\\NN_DoomMarine\\doom.1.5708.bmp"),
		readBMP("D:\\Gamedev\\ExpertSystems.AI\\src\\NN_DoomMarine\\doom.2.35619.bmp"),
		readBMP("D:\\Gamedev\\ExpertSystems.AI\\src\\NN_DoomMarine\\doom.3.14159.bmp"),
		readBMP("D:\\Gamedev\\ExpertSystems.AI\\src\\NN_DoomMarine\\doom.3.14159.bmp"),
		readBMP("D:\\Gamedev\\ExpertSystems.AI\\src\\NN_DoomMarine\\doom.-2.35619.bmp"),
		readBMP("D:\\Gamedev\\ExpertSystems.AI\\src\\NN_DoomMarine\\doom.-1.5708.bmp"),
		readBMP("D:\\Gamedev\\ExpertSystems.AI\\src\\NN_DoomMarine\\doom.-0.785398.bmp");
	Y /= 255.f;

	X << 0.f,
		0.785398f,
		1.5708f,
		2.35619f,
		3.14159f,
		-3.14159f,
		-2.35619f,
		-1.5708f,
		-0.785398f;
	X /= Pi32;

	X = BuildPolynomials(X);

	time(&startTime);
	if(RegisterClassA(&winClass)) {
		HWND window = CreateWindowExA(0, winClass.lpszClassName, "NNet||",
									  WS_OVERLAPPED | WS_SYSMENU | WS_VISIBLE, CW_USEDEFAULT, CW_USEDEFAULT,
									  WINWIDTH * 12, WINHEIGHT * 12, 0, 0, Instance, 0);
		
		neural = Net(int(X.rows()), { 256 }, int(Y.rows()), { ReLU, Sigmoid });
		trainer = NetTrainer(&neural, &X, &Y, 0.15f, 1.f, 0.f);

		HDC deviceContext = GetDC(window);
		vector<float> history;
		vector<float> testHistory;

		int steps = 0;
		//Main Loop
		while(globalRunning) {
			Win32ProcessPendingMessages();
			if(isTraining) {
				trainer.UpdateSingleStep();
				UpdateHistory(history, trainer.GetCache().cost);
				//UpdateHistory(testHistory, trainer.CalcCost(neural.ForwardPropagation(testX), testY));
				steps++;
			} else {
			}
			UpdateDisplay(X, Y, history, testHistory);
			Win32DisplayBufferInWindow(deviceContext, window, backBuffer);
			UpdateWinTitle(steps, window);
		}
		DeleteDC(deviceContext);
	}
	return EXIT_SUCCESS;
}
