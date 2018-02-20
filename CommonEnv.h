#pragma once
#include <Windows.h>
#ifdef __cplusplus
#include <stdexcept>
#endif


typedef float numtype;

#define DEBUG_DEFAULT_PATH "C:/temp/logs"

#ifndef EXPORT
#define EXPORT __declspec(dllexport)
#endif

#define USE_ORCL
//#define USE_GPU

