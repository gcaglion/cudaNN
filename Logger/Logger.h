#pragma once

#include "../CommonEnv.h"
#include "../SharedUtils/SharedUtils.h"
#ifdef USE_ORCL
#include "../OraUtils/OraUtils.h"
#endif
#ifdef USE_GPU
#include "../MyCU/MyCU.h"
#endif

// Logs Destinations
#define PERSIST_TO_TEXT   1
#define PERSIST_TO_ORCL	  2

typedef struct sLogger {
	int dest;
	tDbg* dbg;
	tDBConnection* db;
	tFileData* file;
	bool saveNothing;
	bool saveClient;
	bool saveMSE;
	bool saveRun;
	bool saveInternals;
	bool saveImage;

	EXPORT sLogger(tDBConnection* logDB, bool saveNothing_=false, bool saveClient_=true, bool saveMSE_=true, bool saveRun_=true, bool saveInternals_=false, bool saveImage_=true, tDbg* dbg_=nullptr);
	EXPORT sLogger(tFileData* logFile);
	EXPORT ~sLogger();

	EXPORT void SaveMSE(int pid, int tid, int mseCnt, numtype* mseT, numtype* mseV);
	EXPORT void SaveRun(int pid, int tid, int setid, int npid, int ntid, int runCnt, int featuresCnt, int* feature, numtype* prediction, numtype* actual);
	EXPORT void SaveW(int pid, int tid, int epoch, int Wcnt, numtype* W);
	EXPORT void LoadW(int pid, int tid, int epoch, int Wcnt, numtype* W);
	EXPORT void SaveClient(int pid, char* clientName, DWORD startTime, DWORD duration, int simulLen, char* simulStart, bool doTrain, bool doTrainRun, bool doTestRun);
	EXPORT void Commit();

} tLogger;