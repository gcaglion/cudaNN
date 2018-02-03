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

EXPORT void SBF2BFS_orig(int db, int ds, int dbar, int df, numtype* iv, numtype* ov);

typedef struct sTS {

	tDebugInfo* DebugParms;

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
		dmin=(numtype*)malloc(featuresCnt*sizeof(numtype));
		dmax=(numtype*)malloc(featuresCnt*sizeof(numtype));
		for (int f=0; f<featuresCnt; f++) {
			dmin[f]=1e8; dmax[f]=-1e8;
		}
		scaleM=(numtype*)malloc(featuresCnt*sizeof(numtype));
		scaleP=(numtype*)malloc(featuresCnt*sizeof(numtype));
		dtime=(char**)malloc(len*sizeof(char*)); for (int i=0; i<len; i++) dtime[i]=(char*)malloc(12+1);
		bdtime=(char*)malloc(12+1);
		d=(numtype*)malloc(len*sizeof(numtype));
		bd=(numtype*)malloc(featuresCnt*sizeof(numtype));
		d_tr=(numtype*)malloc(len*sizeof(numtype));
		d_trs=(numtype*)malloc(len*sizeof(numtype));
	}

	~sTS() {
		free(d);
		free(bd);
		free(d_trs);
		free(d_tr);
		for (int i=0; i<len; i++) free(dtime[i]);
		free(dtime); free(bdtime);
	}
	
	EXPORT int load(tFXData* tsFXData, char* pDate0);
	EXPORT int load(tFileData* tsFileData, char* pDate0);
	EXPORT int load(tMT4Data* tsMT4Data, char* pDate0);

	EXPORT int transform(int dt_);
	EXPORT int scale(numtype scaleMin_, numtype scaleMax_);

	EXPORT int TrS(int dt_, numtype scaleMin_, numtype scaleMax_);
	EXPORT int unTrS(numtype scaleMin_, numtype scaleMax_);

	EXPORT int dump(char* dumpFileName="C:/temp/TSdump.csv");

	EXPORT int calcTSF();
	EXPORT int createFromTS(sTS* sourceTS, int* feature);

	EXPORT int buildRunData();

private:
	int LoadOHLCVdata(char* date0);

} TS;

typedef struct sDataSet {
	TS* sourceTS;
	int sampleLen;
	int targetLen;
	int selectedFeaturesCnt;
	int* selectedFeature;

	int samplesCnt;
	int sampleSize;
	int targetSize;
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
	//-- one-step only target+prediction (required by run() ) ???????
	numtype* target0=nullptr;
	numtype* prediction0=nullptr;

	sDataSet(sTS* sourceTS_, int sampleLen_, int targetLen_, int selectedFeaturesCnt_, int* selectedFeature_, int batchSamplesCnt_){
		sourceTS=sourceTS_;
		selectedFeaturesCnt=selectedFeaturesCnt_; selectedFeature=selectedFeature_;
		sampleLen=sampleLen_; sampleSize=sampleLen*selectedFeaturesCnt;
		targetLen=targetLen_; targetSize=targetLen*selectedFeaturesCnt;
		samplesCnt=sourceTS->steps-sampleLen;
		batchSamplesCnt=batchSamplesCnt_;
		batchCnt=(int)floor(samplesCnt/batchSamplesCnt);

		sample=(numtype*)malloc(samplesCnt*sampleSize*sizeof(numtype));
		target=(numtype*)malloc(samplesCnt*targetSize*sizeof(numtype));
		prediction=(numtype*)malloc(samplesCnt*targetSize*sizeof(numtype));
		sampleBFS=(numtype*)malloc(samplesCnt*sampleSize*sizeof(numtype));
		targetBFS=(numtype*)malloc(samplesCnt*targetSize*sizeof(numtype));
		predictionBFS=(numtype*)malloc(samplesCnt*targetSize*sizeof(numtype));
		//--
		target0=(numtype*)malloc(samplesCnt*selectedFeaturesCnt*sizeof(numtype));
		prediction0=(numtype*)malloc(samplesCnt*selectedFeaturesCnt*sizeof(numtype));

		//-- fill sample/target data right at creation time. TS has data in SBF format
		if (buildFromTS(sourceTS)!=0) throw "buildFromTS() failed\n";
		//-- populate BFS sample/target, too
		SBF2BFS();

	}
	~sDataSet() {
		free(sample);
		if (target!=nullptr) free(target);
		free(prediction);
		free(sampleBFS);
		free(targetBFS);
		free(predictionBFS);
		free(target0);
		free(prediction0);
	}

	bool isSelected(int ts_f);
	EXPORT int buildFromTS(sTS* ts);
	EXPORT void SBF2BFS();	//-- fills sampleBFS/targetBFS from sample/target
	EXPORT void BFS2SBF(int vlen, numtype* fromBFS, numtype* toSBF);	//-- fills sample/target from sampleBFS/targetBFS
	EXPORT void dump(char* filename=nullptr);

} DataSet;