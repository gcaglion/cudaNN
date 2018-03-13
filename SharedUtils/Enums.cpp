#include "Enums.h"

//-- in each <class>_enums.h
enumdeclare(SOURCE_DATA, FXDB_SOURCE, FILE_SOURCE, MT4_SOURCE);
//enumdeclare( "SOURCE_DATA", "FXDB_SOURCE", "FILE_SOURCE", "MT4_SOURCE");

//--

bool dd2(char* enumNameS, char* stringToCheck,  int pOptionsCnt, ...) {
	
	va_list			option;
	va_start(option, pOptionsCnt);
	char* option_s;
	bool ret=false;

	for (int o=0; o<pOptionsCnt; o++) {
		option_s= va_arg(option, char*);
		if (strcmp(stringToCheck, option_s)==0) {
			ret=true;
		}
	}
	va_end(option);
	return ret;
}

#define setOptionsCnt(enumName) int optionsCnt=enumName(cnt);

bool dd3m(enumName, stringToCheck, ... ) {
	va_list	option; 
	char* option_s; 
	bool found=false; 
	va_start(option, optionsCnt); 
	printf("cnt=%dn",(cnt)); 
	for(int o=0; o<(cnt); o++){ 
		option_s= va_arg(option, char*); 
		printf("option_s=%sn", option_s); 
		if (strcmp(stringToCheck, option_s)==0) { 
			found=true; 
		} 
	} 
	va_end(option); 
}

EXPORT bool decode(char* paramName, char* stringToCheck, int* oCode, ...) {

	setOptionsCnt(SOURCE_DATA);

	dd3m(SOURCE_DATA, "MT4_SOURCE");


	return true;
}

