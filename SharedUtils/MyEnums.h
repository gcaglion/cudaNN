#pragma once
/*#include "../CommonEnv.h"
#include "Generic.h"
#include "XMLstuff.h"
*/
#include "DataSource_enums.h"
#include "DBConnection_enums.h"
#include "Debugger_enums.h"
#include "FileData_enums.h"
#include "FileInfo_enums.h"
#include "FXData_enums.h"
#include "../TimeSerie/TimeSerie_enums.h"

/*
typedef struct sEnum {
	char*  name;
	int optionsCnt;
	char** option;

	sEnum(...) {
		va_list arguments; int argsCnt;
		va_start(arguments, argsCnt);

		name=va_arg(arguments, char*);
		optionsCnt=argsCnt-1;
		option=(char**)malloc(optionsCnt*sizeof(char*));
		for (int o=0; o<optionsCnt; o++) {
			option[o]=va_arg(arguments, char*);
		}
		va_end(arguments);

	}
	~sEnum() {
		for (int o=0; o<optionsCnt; o++) free(option[o]);
		free(option);
		free(name);
	}

	int decode(char* pName, char* stringToCheck, int* oVal){
		return -1;
	}

} tEnum;

EXPORT bool decode(char* stringToCheck, tParm* enumParm, int* oValsCnt, int* oVal);
*/