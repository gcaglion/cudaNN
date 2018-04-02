#include "ParamMgr.h"

#include "DataSource_enums.h"
#include "DBConnection_enums.h"
#include "Debugger_enums.h"
#include "FileData_enums.h"
#include "FileInfo_enums.h"
#include "FXData_enums.h"
#include "../TimeSerie/TimeSerie_enums.h"
#include "../Logger/Logger_enums.h"
#include "../MyEngines/Engine_enums.h"
#include "../MyEngines/Core_enums.h"

#define optionLookup(option, e) { \
	if (strcmp(parmVal[foundParmId][e], #option)==0) { \
		(*oVal)=option; \
		return true; \
	} \
}
bool sParmsSource::decode(int elementId, int* oVal) { 
	optionLookup(FXDB_SOURCE, elementId);
	optionLookup(FILE_SOURCE, elementId);
	optionLookup(MT4_SOURCE, elementId);
	optionLookup(ORCL_DEST, elementId);
	optionLookup(FILE_DEST, elementId);
	optionLookup(DBG_LEVEL_ERR, elementId);
	optionLookup(DBG_LEVEL_STD, elementId);
	optionLookup(DBG_LEVEL_DET, elementId);
	optionLookup(DBG_DEST_FILE, elementId);
	optionLookup(DBG_DEST_SCREEN, elementId);
	optionLookup(DBG_DEST_BOTH, elementId);
	optionLookup(COMMA_SEPARATOR, elementId);
	optionLookup(TAB_SEPARATOR, elementId);
	optionLookup(SPACE_SEPARATOR, elementId);
	optionLookup(FILE_MODE_READ, elementId);
	optionLookup(FILE_MODE_WRITE, elementId);
	optionLookup(FILE_MODE_APPEND, elementId);
	optionLookup(FXOPEN, elementId);
	optionLookup(FXHIGH, elementId);
	optionLookup(FXLOW, elementId);
	optionLookup(FXCLOSE, elementId);
	optionLookup(FXVOLUME, elementId);
	optionLookup(DT_NONE, elementId);
	optionLookup(DT_DELTA, elementId);
	optionLookup(DT_LOG, elementId);
	optionLookup(DT_DELTALOG, elementId);
	optionLookup(TSF_MEAN, elementId);
	optionLookup(TSF_MAD, elementId);
	optionLookup(TSF_VARIANCE, elementId);
	optionLookup(TSF_SKEWNESS, elementId);
	optionLookup(TSF_KURTOSIS, elementId);
	optionLookup(TSF_TURNINGPOINTS, elementId);
	optionLookup(TSF_SHE, elementId);
	optionLookup(TSF_HISTVOL, elementId);
	optionLookup(ENGINE_XIE, elementId);
	optionLookup(ENGINE_WNN, elementId);
	optionLookup(ENGINE_CUSTOM, elementId);
	optionLookup(CORE_NN, elementId);
	optionLookup(CORE_GA, elementId);
	optionLookup(CORE_SVM, elementId);
	optionLookup(CORE_SOM, elementId);
	optionLookup(CONN_DENSE, elementId);
	optionLookup(CONN_LINEAR, elementId);
	optionLookup(CONN_TRANSFORM, elementId);

	return false;
}