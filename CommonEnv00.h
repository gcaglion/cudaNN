#pragma once

#include <Windows.h>

#ifndef EXPORT
#define EXPORT __declspec(dllexport)
#endif

#ifdef __cplusplus
#include <stdexcept>
#endif


typedef float numtype;
#define DATE_FORMAT "YYYYMMDDHHMI"
#define DATE_FORMAT_LEN 12

#define DEBUG_DEFAULT_PATH "C:/temp/logs"

#define USE_ORCL
//#define USE_GPU

//#define delete obj) if(obj!=nullptr) delete obj