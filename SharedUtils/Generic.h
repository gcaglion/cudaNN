#pragma once
#include "../CommonEnv.h"

EXPORT char* MyGetCurrentDirectory();
EXPORT void Trim(char* str);
EXPORT int cslToArray(const char* csl, char Separator, char** StrList);
EXPORT char* substr(char* str, int start, int len);
EXPORT char* right(char* str, int len);
EXPORT char* left(char* str, int len);
EXPORT void UpperCase(char* str);
EXPORT void removeQuotes(char* istr, char* ostr);
