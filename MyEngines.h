#pragma once

//-- Forecast Engines
#define ENGINE_NN	 0
#define ENGINE_GA	 1
#define ENGINE_SVM	 2
#define ENGINE_SOM	 3
#define ENGINE_WNN   4
#define ENGINE_XIE	 5
//-- Core types
#define CORE_NN 0
#define CORE_GA 1
#define CORE_SVM 2
#define CORE_SOM 3

typedef struct sCore {
	int   CoreType;
	void* CoreSpecs;	// [pointer-to <XXX>_Parms]
	//tCoreLog* CoreLog;		// [Dataset]

	int MSECount;
	int TimeStepsCount;
	int SampleLen;
	int TargetLen;
	int RunCount;
} tCore;

//-- Engine Architecture
typedef struct sEngineDef {
	int EngineType;
	int InputCount;
	int OutputCount;
	int LayersCount;
	int* CoresCount;
	tCore** Core;	// [LayerId][CoreId]
	int TotalCoresCount;
	int TSFcnt;
	int* TSFid;
	int WNN_DecompLevel;
	char WNN_WaveletType[30];
	int AdderCount;	// Training Sessions Counter (ADD_SAMPLE)
#ifdef __cplusplus
	sEngineDef() {
//		TSFid = MallocArray<int>(MAX_TSFCOUNT);
		WNN_DecompLevel = 0;
		strcpy_s(WNN_WaveletType, 29, "Unused");
	}

	~sEngineDef() {
	//	free(TSFid);
	}
#endif

} tEngineDef;

