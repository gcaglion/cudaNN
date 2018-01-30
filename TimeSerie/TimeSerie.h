#pragma once

#include "../CommonEnv.h"
#include "../MyDebug/mydebug.h"
#include "../DBConnection.h"
#include "../fxdata.h"
#include "../filedata.h"
#include "../MT4data.h"

#ifdef USE_ORCL
#include "../OraUtils/OraUtils.h"
#endif // USE_ORCL

#ifdef USE_GPU
#include "../MyCU/MyCU.h"
#endif

//-- Source Types
#define SOURCE_DATA_FROM_FXDB 0
#define SOURCE_DATA_FROM_FILE 1
#define SOURCE_DATA_FROM_MT4   2

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

	tDebugInfo* DebugParms;

	int sourceType;
	tFXData* FXData;
	tFileData* FileData;
	tMT4Data* MT4Data;

	int steps;
	int featuresCnt;
	int len;
	int dt;			// data transformation
	numtype* hd;	//-- host   data ( steps X featuresCnt )
	numtype* dd;	//-- device data ( steps X featuresCnt )
	numtype* hbd;	//-- host   base data ( 1 X featuresCnt )
	numtype* dbd;	//-- device base data ( 1 X featuresCnt )

	//-- constructor / destructor
	sTS(int steps_, int featuresCnt_, tDebugInfo* DebugParms_=nullptr) {
		if(DebugParms_==nullptr){
			DebugParms=new tDebugInfo;
			DebugParms->DebugLevel = 2;
			strcpy(DebugParms->fPath, "C:/temp");
			strcpy(DebugParms->fName, "TimeSerie.log");
			DebugParms->PauseOnError = 1;
		} else {
			DebugParms=DebugParms_;
		}
		steps=steps_;
		featuresCnt=featuresCnt_;
		len=steps*featuresCnt;
		hd=(numtype*)malloc(len*sizeof(numtype));
		hbd=(numtype*)malloc(featuresCnt*sizeof(numtype));
		#ifdef USE_GPU
		if (cudaMalloc(&dd, len*sizeof(numtype)!=cudaSuccess)) throw FAIL_CUDAMALLOC;
		if (cudaMalloc(&dbd, featuresCnt*sizeof(numtype)!=cudaSuccess)) throw FAIL_CUDAMALLOC;
		#endif
	}
	~sTS() {
		free(hd);
		free(hbd);
		#ifdef USE_GPU
		cudaFree(dd);
		cudaFree(dbd);
		#endif
	}
	
	EXPORT int load(tFXData* tsFXData, char* pDate0);
	EXPORT int load(tFileData* tsFileData, char* pDate0);
	EXPORT int load(tMT4Data* tsMT4Data, char* pDate0);

	EXPORT int transform();
	EXPORT int untransform();
	EXPORT int scale();
	EXPORT int unscale();
	EXPORT int calcTSF();
	EXPORT int createFromTS(sTS* sourceTS, int* feature);

	EXPORT int buildTrainData();
	EXPORT int buildRunData();

	EXPORT void SBF2BFS(int db, int ds, int dbar, int df, numtype* iSBFv, numtype* oBFSv);
	EXPORT void BFS2SBF(int db, int ds, int dbar, int df, numtype* iBFSv, numtype* oSBFv);

private:
	int LoadOHLCVdata(char* date0);


} TS;