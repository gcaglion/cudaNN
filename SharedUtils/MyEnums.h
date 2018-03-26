#pragma once
#include "../CommonEnv.h"

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