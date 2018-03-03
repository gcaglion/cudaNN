#pragma once

#include "../CommonEnv.h"
#include "../SharedUtils/SharedUtils.h"

#ifdef USE_ORCL
#include "../OraUtils/OraUtils.h"
#endif // USE_ORCL

#ifdef USE_GPU
#include "../MyCU/MyCU.h"
#endif

//-- Data Tranformations
#define DT_NONE		 0
#define DT_DELTA	 1
#define DT_LOG		 2
#define DT_DELTALOG	 3

//-- Statistical Features
#define TSF_MEAN 0
#define TSF_MAD 1
#define TSF_VARIANCE 2
#define TSF_SKEWNESS 3
#define TSF_KURTOSIS 4
#define TSF_TURNINGPOINTS 5
#define TSF_SHE 6
#define TSF_HISTVOL 7

typedef struct sTS {

	tDbg* dbg;

	int sourceType;
	tFXData* FXData;
	tFileData* FileData;
	tMT4Data* MT4Data;

	int steps;
	int featuresCnt;
	int len;
	int dt;						// data transformation
	// data scaling: different parameters for each feature
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
	EXPORT void sTScommon(int steps_, int featuresCnt_, tDbg* dbg_);
	EXPORT sTS(int steps_, int featuresCnt_, tDbg* dbg_=nullptr);
	EXPORT sTS(tFXData* dataSource_, int steps_, char* date0_, int dt_, numtype scaleMin_, numtype scaleMax_, tDbg* dbg_=nullptr);
	EXPORT sTS(tFileData* dataSource_, int steps_, char* date0_, int dt_, numtype scaleMin_, numtype scaleMax_, tDbg* dbg_=nullptr);
	EXPORT sTS(tMT4Data* dataSource_, int steps_, char* date0_, int dt_, numtype scaleMin_, numtype scaleMax_, tDbg* dbg_=nullptr);
	EXPORT ~sTS();
	
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

} tTS;

typedef struct sDataSet {
	tDbg* dbg;

	tTS* sourceTS;
	int sampleLen;
	int targetLen;
	int selectedFeaturesCnt;
	int* selectedFeature;

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
	EXPORT sDataSet(sTS* sourceTS_, int sampleLen_, int targetLen_, int selectedFeaturesCnt_, int* selectedFeature_, int batchSamplesCnt_, tDbg* dbg_=nullptr);
	EXPORT ~sDataSet();

	bool isSelected(int ts_f);
	EXPORT void buildFromTS(tTS* ts);
	EXPORT void SBF2BFS(int batchId, int barCnt, numtype* fromSBF, numtype* toBFS);
	EXPORT void BFS2SBF(int batchId, int barCnt, numtype* fromBFS, numtype* toSBF);
	EXPORT void BFS2SFB(int batchId, int barCnt, numtype* fromBFS, numtype* toSFB);
	EXPORT void BFS2SFBfull(int barCnt, numtype* fromBFS, numtype* toSFB);
	EXPORT void dump(char* filename=nullptr);

} tDataSet;