#pragma once
#include "../CommonEnv.h"

#include "DataSource_enums.h"
#include "DBConnection_enums.h"
#include "DebugInfo_enums.h"
#include "FileData_enums.h"
#include "FileInfo_enums.h"
#include "FXData_enums.h"
#include "../TimeSerie/TimeSerie_enums.h"
#include "../cuNN/NN_parms.h"

EXPORT int decode(char* paramName, char* stringToCheck, int* oCode);