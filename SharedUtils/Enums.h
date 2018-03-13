#pragma once
#include "../CommonEnv.h"

#include "DataSource_enums.h"
#include "DBConnection_enums.h"
#include "Debugger_enums.h"
#include "FileData_enums.h"
#include "FileInfo_enums.h"
#include "FXData_enums.h"
#include "../TimeSerie/TimeSerie_enums.h"
#include "../cuNN/NN_parms.h"


typedef struct sDecoder {

} tDecoder;

EXPORT bool decode(char* paramName, char* stringToCheck, int* oCode, ...);

//-- in Enums.h
#define enumdeclare(name, ...) enum name { __VA_ARGS__, cnt }; 
//--

