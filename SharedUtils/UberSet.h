#pragma once
#include "DebugInfo.h"
#include "ParamMgr.h"
#include "../TimeSerie/TimeSerie.h"
#include "../MyEngines/CoresParms.h"

//-- predefined Uber-Sets
#define US_MODEL 0
#define US_TRAIN 1
#define US_TEST  2
#define US_VALID 3
#define US_CORE_NN  10
#define US_CORE_SVM 11
#define US_CORE_GA  12

typedef struct sUberSetParms {

	tDbg* dbg;
	tParamMgr* parms;

	//-- NN Core parms
	tNNparms* NN;

	//-- 0. Data model parms (set-invariant)
	int SampleLen;
	int PredictionLen;
	int FeaturesCnt;

	//-- 1. common parameters:
	bool doIt;
	//-- datasource-independent timeserie properties
	tTimeSerie* TS;
	char TSdate0[13]; int TShistoryLen; int TS_DT; bool TS_BWcalc;
	//-- data source
	int TS_DS_type;
	//-- FXDB-type data source properties
	tFXData* TS_DS_FX; tDBConnection* TS_DS_FX_DB; char TS_DS_FX_DBUser[DBUSER_MAXLEN]; char TS_DS_FX_DBPassword[DBPASSWORD_MAXLEN]; char TS_DS_FX_DBConnString[DBCONNSTRING_MAXLEN];
	char TS_DS_FX_Symbol[FX_SYMBOL_MAX_LEN]; char TS_DS_FX_TimeFrame[FX_TIMEFRAME_MAX_LEN]; bool TS_DS_FX_IsFilled;
	//-- File-type data source properties
	tFileData* TS_DS_File; char TS_DS_File_FullName[MAX_PATH]; int TS_DS_File_FieldSep; int* TS_DS_File_BWcol=new int(2);
	//-- dataset properties
	tDataSet* DataSet; int BatchSamplesCnt; int SelectedFeaturesCnt; int* SelectedFeature;

	//-- 2. TRAIN-specific parameters
	bool doTrainRun;
	//-- 3. TEST-specific parameters
	bool runFromSavedNet; int testWpid; int testWtid;

	EXPORT sUberSetParms(tParamMgr* parms_, int set, tDbg* dbg_);

} tUberSetParms;

