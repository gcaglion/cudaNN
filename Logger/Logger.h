#pragma once

#include "../CommonEnv.h"
#include "../MyDebug/mydebug.h"
#ifdef USE_ORCL
#include "../OraUtils/OraUtils.h"
#endif
#include "../DBConnection.h"
#include "../DataFile.h"
#ifdef USE_GPU
#include "../MyCU/MyCU.h"
#endif

// Logs Destinations
#define LOG_TO_TEXT   1
#define LOG_TO_ORCL	  2
/*
EXPORT int LogSaveMSE(tDebugInfo* DebugParms, int pid, int tid, int mseCnt, numtype* mseT, numtype* mseV);
EXPORT int LogSaveRun(tDebugInfo* DebugParms, int pid, int tid, int runCnt, int featuresCnt, numtype* prediction, numtype* actual);
EXPORT int LogSaveW(tDebugInfo* DebugParms, int pid, int tid, int epoch, int Wcnt, numtype* W);
EXPORT int LogLoadW(tDebugInfo* DebugParms, int pid, int tid, int epoch, int Wcnt, numtype* W);
EXPORT int LogSaveClient(tDebugInfo* DebugParms, int pid, char* clientName, DWORD startTime, DWORD duration, int simulLen, char* simulStart, int doTraining, int doRun);
EXPORT void Commit(tDebugInfo* DebugParms);
*/
typedef struct sLogger {
	int dest;
	tDebugInfo* dbg;
	tDBConnection* db;
	tDataFile* file;
	bool saveNothing=false;
	bool saveClient=true;
	bool saveMSE=true;
	bool saveRun=true;
	bool saveInternals=false;
	bool saveImage=true;

#ifdef __cplusplus
	EXPORT sLogger(tDebugInfo* DebugProps, tDBConnection* logDB) {
		dbg=DebugProps;
		dest=LOG_TO_ORCL;
		db=logDB;
	}
	EXPORT sLogger(tDataFile* logFile) {
		dest=LOG_TO_TEXT;
		file=logFile;
	}
	EXPORT ~sLogger() {}

	EXPORT bool SaveMSE(int pid, int tid, int mseCnt, numtype* mseT, numtype* mseV);
	EXPORT bool SaveRun(int pid, int tid, int runCnt, int featuresCnt, numtype* prediction, numtype* actual);
	EXPORT bool SaveW(int pid, int tid, int epoch, int Wcnt, numtype* W);
	EXPORT bool LoadW(int pid, int tid, int epoch, int Wcnt, numtype* W);
	EXPORT bool SaveClient(int pid, char* clientName, DWORD startTime, DWORD duration, int simulLen, char* simulStart, int doTraining, int doRun);
	EXPORT void Commit();

#endif

} tLogger;