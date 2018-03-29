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

typedef struct sTimeSerie {

	tDebugger* dbg;

	//-- data set
	int set;
	//-- data source
	int sourceType;
	tFXData* fxData;
	tFileData* fileData;
	tMT4Data* mt4Data;

	int steps;
	int featuresCnt;
	int len;
	int dt;	// data transformation

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
	EXPORT void sTimeSeriecommon(int steps_, int featuresCnt_, tDebugger* dbg_);
	EXPORT sTimeSerie(int steps_, int featuresCnt_, tDebugger* dbg_=nullptr);
	EXPORT sTimeSerie(tFXData* dataSource_, int steps_, char* date0_, int dt_, tDebugger* dbg_=nullptr);
	EXPORT sTimeSerie(tFileData* dataSource_, int featuresCnt_, int steps_, char* date0_, int dt_, tDebugger* dbg_=nullptr);
	EXPORT sTimeSerie(tMT4Data* dataSource_, int steps_, char* date0_, int dt_, tDebugger* dbg_=nullptr);
	EXPORT sTimeSerie(tParmsSource* parms, int set_, tDebugger* dbg_);
	EXPORT ~sTimeSerie();
	
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

typedef struct sDataSet {
	tDebugger* dbg;

	tTimeSerie* sourceTS;
	int sampleLen;
	int targetLen;
	int selectedFeaturesCnt;
	int* selectedFeature;
	int* datafileBWFeature;

	int samplesCnt;
	int batchSamplesCnt;
	int batchCnt;

	//-- sample, target, prediction are stored in  order (Sample-Bar-Feature)
	numtype* sample=nullptr;
	numtype* target=nullptr;
	numtype* prediction=nullptr;
	//-- network training requires BFS ordering
	numtype* sampleBFS=nullptr;
	numtype* targetBFS=nullptr;
	numtype* predictionBFS=nullptr;
	//-- network inference requires SFB ordering to get first-step prediction
	numtype* targetSFB=nullptr;
	numtype* predictionSFB=nullptr;
	//-- one-step only target+prediction (required by run() ) ???????
	numtype* target0=nullptr;
	numtype* prediction0=nullptr;

	//-- constructor / destructor
	EXPORT sDataSet(sTimeSerie* sourceTS_, int sampleLen_, int targetLen_, int batchSamplesCnt_, int selectedFeaturesCnt_, int* selectedFeature_, int* datafileBWFeature_, tDebugger* dbg_=nullptr);
	EXPORT sDataSet(tParmsSource* parms, sTimeSerie* sourceTS_, tDebugger* dbg_);
	EXPORT ~sDataSet();

	bool isSelected(int ts_f);
	EXPORT void buildFromTS(tTimeSerie* ts);
	EXPORT void SBF2BFS(int batchId, int barCnt, numtype* fromSBF, numtype* toBFS);
	EXPORT void BFS2SBF(int batchId, int barCnt, numtype* fromBFS, numtype* toSBF);
	EXPORT void BFS2SFB(int batchId, int barCnt, numtype* fromBFS, numtype* toSFB);
	EXPORT void BFS2SFBfull(int barCnt, numtype* fromBFS, numtype* toSFB);
	EXPORT void dump(char* filename=nullptr);

} tDataSet;