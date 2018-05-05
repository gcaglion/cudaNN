#pragma once

#include <Windows.h>

#include "CommonMacros.h"

#ifndef EXPORT
#define EXPORT __declspec(dllexport)
#endif

#ifdef __cplusplus
#include <stdexcept>
#endif


typedef float numtype;
#define DATE_FORMAT "YYYYMMDDHHMI"
#define DATE_FORMAT_LEN 12

#define USE_ORCL
//#define USE_GPU

#define OBJNAME_MAXLEN 128

