#pragma once
typedef void * sql_context;

#ifdef __cplusplus
#undef EXPORT
#define EXPORT extern "C" __declspec(dllexport)
#else
typedef int bool;
#define true 1
#define false 0
#endif

#include "../CommonEnv.h"
#include "../fxdata.h"
#include "../MyDebug/mydebug.h"
#include <math.h>



//=== DB common functions
EXPORT bool OraConnect(tDebugInfo* DebugInfo, tDBConnection* DBConnInfo);
EXPORT void OraDisconnect(tDBConnection* DBConnInfo, int Commit);
EXPORT void OraCommit(tDBConnection* DBConnInfo);

//=== Retrieval functions
EXPORT bool Ora_GetFlatOHLCV(tDebugInfo* DebugParms, tDBConnection* db, char* pSymbol, char* pTF, char* pDate0, int pRecCount, char** oBarTime, float* oBarData, char* oBaseTime, float* oBaseBar);

//=== Logging functions
EXPORT bool Ora_LogSaveMSE(tDebugInfo* DebugParms, tDBConnection* db, int pid, int tid, int mseCnt, float* mseT, float* mseV);
EXPORT bool Ora_LogSaveW(tDebugInfo* DebugParms, tDBConnection* db, int pid, int tid, int epoch, int Wcnt, numtype* W);
EXPORT bool Ora_LogSaveClient(tDebugInfo* DebugParms, tDBConnection* db, int pid, char* clientName, DWORD startTime, DWORD duration, int simulLen, char* simulStart, int doTraining, int doRun);
EXPORT bool Ora_LogSaveRun(tDebugInfo* DebugParms, tDBConnection* db, int pid, int tid, int barCnt, int featuresCnt, numtype* prediction, numtype* actual);
EXPORT bool Ora_LogLoadW(tDebugInfo* DebugParms, tDBConnection* db, int pid, int tid, int epoch, int Wcnt, numtype* oW);

#ifdef __cplusplus
#undef EXPORT
#define EXPORT __declspec(dllexport)
#endif

