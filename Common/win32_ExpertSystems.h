#pragma  once
#if !defined(WIN32_DEEPLEARNING_H)
#include <math.h>
#include <stdint.h>
#include <windows.h>
#include <stdio.h>
#include <malloc.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <Eigen/dense>
#include "stdDraw.h"
#include "stdMat.h"
#include "stdNet.h"


#define internal static 
#define local_persist static 
#define global_variable static

using namespace Eigen;

typedef int8_t int8;
typedef int16_t int16;
typedef int32_t int32;
typedef int64_t int64;
typedef int32 bool32;

typedef uint8_t uint8;
typedef uint16_t uint16;
typedef uint32_t uint32;
typedef uint64_t uint64;

typedef float real32;
typedef double real64;

#define Assert(Expression) if(!(Expression)) {*(int *)0 = 0;}

#define WIN32_DEEPLEARNING_H
#endif