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

//-- generic (non-classed)
EXPORT void Trim(char* str);
EXPORT int cslToArray(char* csl, char Separator, char** StrList);
