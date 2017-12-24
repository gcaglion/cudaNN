#pragma once

#define _CRT_RAND_S
#include "../CommonEnv.h"
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdio.h>
#include <direct.h>

#include "MyUtilsT.cpp"

//------------ CPU Usage utilities -------------------
EXPORT float GetCPULoad();
EXPORT DWORD GetMemLoad();

//------	Random Utilities	- Start		--------------
//#define MyRnd(rmin, rmax) (rmin + (rand() % rmax))

EXPORT int __stdcall	MyRndInt(int rmin, int rmax);
EXPORT numtype __stdcall MyRndDbl(numtype min, numtype max);
EXPORT numtype		__stdcall MyRandom01();
//------	Random Utilities	- End		--------------

//------	Time Utilities	- Start		--------------
EXPORT int		__stdcall TimeFrameToMins(char* tf);
EXPORT char*	__stdcall timestamp();
EXPORT int		__stdcall TimeFrameToSecs(char* tf);
EXPORT time_t	__stdcall TimeStringToTime(char* TimeString);
EXPORT void		__stdcall TimeToTimeString(time_t pTime, char* pcTime);
EXPORT void		__stdcall ms2ts(numtype ims, char* ots);
//------	Time Utilities	- End		--------------

//------	String Utilities	- Start		--------------
#define BUFSIZE MAX_PATH

EXPORT void __stdcall StringSort(int strcnt, char** str);
EXPORT char* __stdcall substr(char* str, int start, int len);
EXPORT char* __stdcall right(char* str, int len);
EXPORT char* __stdcall left(char* str, int len);
EXPORT void __stdcall Trim(char* str);
EXPORT void __stdcall UpperCase(char* str);
EXPORT wchar_t *convertCharArrayToLPCWSTR(const char* charArray);
EXPORT char*    convertLPCWSTRToCharArray(LPCWSTR wideStr);
EXPORT wchar_t* FullFileName(char* pPath, LPCWSTR pFilename);
EXPORT char* MyGetCurrentDirectory();

//------	String Utilities	- End		--------------

//------	Arrays Utilities	- Start		--------------
#define SHIFT_FORWARD 1
#define SHIFT_BACKWARD 2

EXPORT void	__stdcall ShuffleArray(int *array, size_t n);
EXPORT void	__stdcall ShiftArray(int direction, int ArrLen, numtype* Arr, numtype NewVal);
EXPORT void	__stdcall InvertArray(int ArrLen, numtype* Arr);
EXPORT int	__stdcall DumpArrayD(int ArrLen, numtype* Arr, char* fname);

//------	Arrays Utilities	- End	--------------

//------	Miscellaneous Utilities	- Start	----------
//#define sign(a) ( ( (a) < 0 )  ?  -1   : ( (a) > 0 ) )
#define greater(a,b) ((a>b)?a:b)
EXPORT int			__stdcall FindMinMax(int pDataCount, numtype* pData, numtype* oMin, numtype* oMax);

EXPORT int cslToArray(char* csl, char Separator, char** StrList);

EXPORT int __stdcall	ReadMultiParamFromFileI(char* pFileName, char* pParamName, int** oParamValue);
//----------------------------------------------


EXPORT void			__stdcall swapD(numtype* a, numtype* b);
EXPORT void			__stdcall swapI(int* a, int* b);
EXPORT void __stdcall gotoxy(int x, int y);
EXPORT int isIndeterminate(const numtype pV);
//------	Miscellaneous Utilities	- End	----------

//------	Data Scaling Utilities			---------
#define EMPTY_VALUE -9999	// This is needed for partial Unscaling and UnTransformation of Dataseries

EXPORT void __stdcall SlideArray(int iWholeSetLen, numtype* iWholeSet, int iSampleCount, int iSampleLen, numtype** oSample, int iTargetLen, numtype** oTarget, int pWriteLog);
EXPORT void __stdcall SlideArrayNew(int dlen, numtype* idata, int iSampleLen, int iTargetLen, numtype** oSample, numtype** oTarget, int pWriteLog);
EXPORT void __stdcall UnSlideArray(int pRowsCnt, int pSampleLen, int pTargetLen, numtype** pSample, numtype** pTarget, numtype* oArr);
EXPORT void __stdcall  DataScale(int iDataLen, numtype* iOrigData, numtype iScaleMin, numtype iScaleMax, numtype* oScaledData, numtype* oScaleM, numtype* oScaleP);
EXPORT void __stdcall  DataScale(int iDataLen, numtype* iOrigData, numtype iScaleMin, numtype iScaleMax, numtype* oScaledData, numtype iScaleM, numtype iScaleP);		//-- overloaded version used to scale an array using existing M/P
EXPORT void __stdcall  DataUnScale(int iDataLen, int iFromItem, int iToItem, numtype* iScaledData, numtype iScaleM, numtype iScaleP, numtype* oUnScaledData);
//------	Data Scaling Utilities - End	---------
