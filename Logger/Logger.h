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
#define LOG_TO_TEXT   1
#define LOG_TO_ORCL	  2

typedef struct sLogger {
	int dest;
	tDbg* dbg;
	tDBConnection* db;
	tDataFile* file;
	bool saveNothing=false;
	bool saveClient=true;
	bool saveMSE=true;
	bool saveRun=true;
	bool saveInternals=false;
	bool saveImage=true;

	EXPORT sLogger(tDBConnection* logDB, tDbg* dbg_);
	EXPORT sLogger(tDataFile* logFile);
	EXPORT ~sLogger();

	EXPORT bool SaveMSE(int pid, int tid, int mseCnt, numtype* mseT, numtype* mseV);
	EXPORT bool SaveRun(int pid, int tid, int runCnt, int featuresCnt, numtype* prediction, numtype* actual);
	EXPORT bool SaveW(int pid, int tid, int epoch, int Wcnt, numtype* W);
	EXPORT bool LoadW(int pid, int tid, int epoch, int Wcnt, numtype* W);
	EXPORT bool SaveClient(int pid, char* clientName, DWORD startTime, DWORD duration, int simulLen, char* simulStart, bool doTraining, bool doRun);
	EXPORT void Commit();

} tLogger;