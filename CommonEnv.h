#pragma once
#include <Windows.h>
#ifdef __cplusplus
#include <stdexcept>
#endif


typedef float numtype;

//-- these are used for both read (SourceData) and write (Logger) operations
#define ORCL 1
#define TXT 2

#define DEBUG_DEFAULT_PATH "C:/temp/logs"

#ifndef EXPORT
#define EXPORT __declspec(dllexport)
#endif

#define USE_ORCL
//#define USE_GPU

