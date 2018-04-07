#pragma once

#include "../CommonEnv00.h"

EXPORT char* MyGetCurrentDirectory();
EXPORT bool getCurrentPath(char* oPath);
EXPORT int cslToArray(const char* csl, char Separator, char** StrList);
EXPORT char* substr(char* str, int start, int len);
EXPORT char* right(char* str, int len);
EXPORT char* left(char* str, int len);
EXPORT int  instr(char soughtChar, char* intoStr, bool fromRight);
EXPORT void UpperCase(char* str);
EXPORT void removeQuotes(char* istr, char* ostr);
EXPORT void stripChar(char* istr, char c);
EXPORT bool getValuePair(char* istr, char* oName, char* oVal, char eqSign);
EXPORT bool isnumber(char* str);
EXPORT bool isInList(int soughtVal, int listLen, int* list);
