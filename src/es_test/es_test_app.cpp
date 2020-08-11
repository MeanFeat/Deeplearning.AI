#include "es_test_app.h"

static vector<string> headers;
static vector<string> prefixes;
static vector<string> functionNames;
static vector<vector<vector<int>>> arguments;

void BuildTempTests(const string fSource, const string fDest, int lineNum){
#undef TEST_TEMP_LISTS
	std::string line;
	ifstream source(fSource);
	ofstream dest(fDest.c_str());
	dest.clear();
	int i = 0;
	while (source.good()) {
		std::getline(source, line);
		if (i == lineNum) {
			dest << line << endl;
			break;
		}
		i++;
	}
	dest.close();
	source.close();
}

void ReadTestList(const string fName) {
	std::string line;
	ifstream file(fName);
	ParseState state = ParseState::none;
	vector<int> parseArgs;
	vector<vector<int>> currentFuncArgs;
	while (file.good()) {
		std::getline(file, line);
		std::stringstream iss(line);
		std::string val, lastVal;
		while (iss.good()) {
			std::getline(iss, val, ':');
			if (state == ParseState::none) {
				if (strFind(lastVal, "header")) {
					state = ParseState::header;
					headers.push_back(val);
				}
				else if (strFind(lastVal, "prefix")) {
					state = ParseState::prefix;
					prefixes.push_back(val);
				}
				else if (strFind(lastVal, "functionName")) {
					state = ParseState::functionName;
					functionNames.push_back(val);
				}
				if (strFind(lastVal, "arguments")) {
					state = ParseState::args;
				}
				lastVal = val;
			}
			switch (state) {
			case ParseState::none:
			case ParseState::header:
			case ParseState::prefix:
			case ParseState::functionName: //fall through
				state = ParseState::none;
				break;
			case ParseState::args: {
				if (strFind(val, "}")) {
					state = ParseState::none;
					arguments.push_back(currentFuncArgs);
					currentFuncArgs.clear();
				}
				else {
					if (!strFind(val, "{")) {
						size_t pos = 0;
						std::string token;
						do {
							token = val.substr(0, pos);
							val.erase(0, pos + 1);
							int temp;
							strCast(&temp, val);
							parseArgs.push_back(temp);
						} while ((pos = val.find(',')) != std::string::npos);
						currentFuncArgs.push_back(parseArgs);
						parseArgs.clear();
					}
				}
			} break;
			default:
				state = ParseState::none;
				break;
			}
		}
	}
	file.close();
}
void CreateGeneratedUnit(const string fileName) {
	ofstream file(fileName.c_str());
	file.clear();
	file << "//GENERATED FILE" << endl;
	for (int fn = 0; fn < functionNames.size(); fn++) {
		string className = headers[fn];
		className.erase(remove_if(className.begin(), className.end(),
			[](unsigned char x) {return x == '\"'; }), className.end());
		file << "TEST_CLASS(" << className << ") { public:" << endl;
		for (int arg = 0; arg < arguments[fn].size(); arg++) {
			string args;
			file << "NAME_RUN(" << prefixes[fn] << "_";
			for (int a = 0; a < arguments[fn][arg].size(); a++) {
				file << arguments[fn][arg][a];
				if (a < arguments[fn][arg].size() - 1) {
					file << "x";
				}
				else {
					file << ",";
				}
			}
			file << functionNames[fn] << "(";
			for (int a = 0; a < arguments[fn][arg].size(); a++) {
				file << arguments[fn][arg][a];
				if (a < arguments[fn][arg].size() - 1) {
					file << ",";
				}
			}
			file << "));" << endl;
		}
		file << "};" << endl;
	}
	file.close();
}
void CreateGeneratedCpp(const string fileName) {
	ofstream file(fileName.c_str());
	file.clear();
	file << "//GENERATED FILE" << endl;
	for (int fn = 0; fn < functionNames.size(); fn++) {
		file << "PrintHeader(" << headers[fn] << ");" << endl;
		for (int arg = 0; arg < arguments[fn].size(); arg++) {
			file << functionNames[fn] << "(";
			for (int a = 0; a < arguments[fn][arg].size(); a++) {
				file << arguments[fn][arg][a];
				if (a < arguments[fn][arg].size() - 1) {
					file << ", ";
				}
				else {
					file << ");" << endl;
				}
			}
		}
	}
	file.close();
}

void RunAllTests() {
	initParallel();
	setNbThreads(4);
	verbosity = 2;
#ifndef TEST_LISTS
#define TEST_LISTS
#include GENERATED_TESTS
#endif
}

int main() {
	cout << "(T)est individual, Test (A)ll, (B)uild, or e(X)it: ";
	HANDLE handle = GetStdHandle(STD_INPUT_HANDLE);
	DWORD events;
	INPUT_RECORD buffer;
	bool pendingTest = false;
	while (1) {
		if (pendingTest){
			pendingTest = false;
			break;
		}
		PeekConsoleInput(handle, &buffer, 1, &events);
		if (events > 0) {
			ReadConsoleInput(handle, &buffer, 1, &events);
			WORD in = buffer.Event.KeyEvent.wVirtualKeyCode;
			if (in == 66) { //'b'
				cout << "Building File" << endl;
				ReadTestList("tests.list");
				CreateGeneratedCpp("test_cpp.generated");
				CreateGeneratedUnit("test_unit.generated");
				break;
			} else if (in == 65) { //'a'
				cout << "Running All Tests" << endl;
				RunAllTests();
			} else if (in == 84) { //'t'
				cout << "Test " << endl;
				ifstream file(GENERATED_TESTS);
				string line;
				int lineNumber = 0;
				while(file.good()){
					std::getline(file, line);
					if (line != "") {
						string spacer = lineNumber <= 9 ? "  " : " ";
						cout << spacer << lineNumber << ": " << line << endl;
						lineNumber++;
					}
				}
				string inTest;
				cin >> inTest;
				int temp;
				strCast(&temp, inTest);
				BuildTempTests(GENERATED_TESTS, GENERATED_TEMP_TESTS, temp);
				pendingTest = true;
			}
			else if (in == 88) { //'x'
				break;
			}
		}
	}
	return 0;
}