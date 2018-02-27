#pragma once
#include "../CommonEnv.h"

#include <stdio.h>
#include <time.h>

#include "DebugInfo.h"
#include "FileInfo.h"
#include "DataFile.h"
#include "DBConnection.h"
#include "FXData.h"
#include "MT4data.h"
#include "ParamMgr.h"

//-- generic (non-classed)
EXPORT char* MyGetCurrentDirectory();
EXPORT void UpperCase(char* str);
EXPORT void Trim(char* str);
EXPORT int cslToArray(char* csl, char Separator, char** StrList);
EXPORT char* substr(char* str, int start, int len);
EXPORT char* right(char* str, int len);
EXPORT char* left(char* str, int len);
EXPORT void UpperCase(char* str);
