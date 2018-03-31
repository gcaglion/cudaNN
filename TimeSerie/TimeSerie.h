#pragma once

#include "../CommonEnv.h"
#include "TimeSerie_enums.h"
#include "../SharedUtils/FXData.h"
#include "../SharedUtils/FileData.h"
#include "../SharedUtils/MT4Data.h"

#ifdef USE_ORCL
#include "../OraUtils/OraUtils.h"
#endif // USE_ORCL

#ifdef USE_GPU
#include "../MyCU/MyCU.h"
#endif

#define MAX_DATA_FEATURES 128
#define MAX_TSF_CNT	32

typedef struct sTimeSerie {

	tDebugger* dbg;

	//-- data source
	int sourceType;
	tFXData* fxData;
	tFileData* fileData;
	tMT4Data* mt4Data;

	char date0[DATE_FORMAT_LEN];
	int steps;
	int featuresCnt;
	int len;
	int dt;	// data transformation
	bool BWcalc;	// Bar width calc
	int tsfCnt;
	int* tsf;

	// data scaling: boundaries depend on core the samples are fed to, M/P are different for each feature
	numtype scaleMin, scaleMax;
	numtype *scaleM, *scaleP;
	numtype *dmin, *dmax;

	numtype* d;		//-- host   data ( steps X featuresCnt )
	char** dtime;	//-- may always be useful...
	numtype* bd;	//-- host   base data ( 1 X featuresCnt )
	char* bdtime;	
	bool hasTR=false;
	numtype* d_tr;
	bool hasTRS=false;
	numtype* d_trs;

	//-- constructors / destructor
	EXPORT void sTimeSeriecommon(int steps_, int featuresCnt_, int tsfCnt_=0, int* tsf_=nullptr, tDebugger* dbg_=nullptr);
	EXPORT sTimeSerie(int steps_, int featuresCnt_, tDebugger* dbg_=nullptr);
	EXPORT sTimeSerie(tFXData* dataSource_, int steps_, char* date0_, int dt_, tDebugger* dbg_=nullptr);
	EXPORT sTimeSerie(tFileData* dataSource_, int featuresCnt_, int steps_, char* date0_, int dt_, tDebugger* dbg_=nullptr);
	EXPORT sTimeSerie(tMT4Data* dataSource_, int steps_, char* date0_, int dt_, tDebugger* dbg_=nullptr);
	EXPORT sTimeSerie(tParmsSource* parms, char* parmKey, tDebugger* dbg_=nullptr);
	EXPORT ~sTimeSerie();
	EXPORT sTimeSerie() {}

	EXPORT void load(tFXData* tsFXData, char* pDate0);
	EXPORT void load(tFileData* tsFileData, char* pDate0);
	EXPORT void load(tMT4Data* tsMT4Data, char* pDate0);

	EXPORT void transform(int dt_);
	EXPORT void scale(numtype scaleMin_, numtype scaleMax_);

	EXPORT void TrS(int dt_, numtype scaleMin_, numtype scaleMax_);
	EXPORT void unTrS(numtype scaleMin_, numtype scaleMax_);

	EXPORT void dump(char* dumpFileName="C:/temp/TSdump.csv");

private:
	bool LoadOHLCVdata(char* date0);

} tTimeSerie;

