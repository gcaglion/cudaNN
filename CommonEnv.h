#pragma once

#include <Windows.h>
#ifdef __cplusplus
#include <stdexcept>

class sBaseObj {
public:
	virtual ~sBaseObj(){
		//printf("sBaseObj() destructor called.\n"); 
	}
};

#endif


typedef float numtype;
#define DATE_FORMAT "YYYYMMDDHHMI"
#define DATE_FORMAT_LEN 12

#define DEBUG_DEFAULT_PATH "C:/temp/logs"

#ifndef EXPORT
#define EXPORT __declspec(dllexport)
#endif

#define USE_ORCL
//#define USE_GPU

#define cleanup(obj) if(obj!=nullptr) delete obj