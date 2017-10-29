// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once

#include "targetver.h"
#include <stdio.h>
#include <tchar.h>
#include "stdMat.h"
#include <chrono>

#define Assert(Expression) if(!(Expression)) {*(int *)0 = 0;}

using namespace Eigen;
using namespace std;
using namespace std::chrono;

// TODO: reference additional headers your program requires here
