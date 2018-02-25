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
#include "../SharedUtils/FXData.h"
#include "../SharedUtils/DBConnection.h"
#include "../SharedUtils/DebugInfo.h"
#include <math.h>



//=== DB common functions
EXPORT bool OraConnect(tDbg* DebugInfo, tDBConnection* DBConnInfo);
EXPORT void OraDisconnect(tDBConnection* DBConnInfo, int Commit);
EXPORT void OraCommit(tDBConnection* DBConnInfo);

//=== Retrieval functions
EXPORT bool Ora_GetFlatOHLCV(tDbg* dbg, tDBConnection* db, char* pSymbol, char* pTF, char* pDate0, int pRecCount, char** oBarTime, float* oBarData, char* oBaseTime, float* oBaseBar);

//=== Logging functions
EXPORT bool Ora_LogSaveMSE(tDbg* dbg, tDBConnection* db, int pid, int tid, int mseCnt, float* mseT, float* mseV);
EXPORT bool Ora_LogSaveW(tDbg* dbg, tDBConnection* db, int pid, int tid, int epoch, int Wcnt, numtype* W);
EXPORT bool Ora_LogSaveClient(tDbg* dbg, tDBConnection* db, int pid, char* clientName, DWORD startTime, DWORD duration, int simulLen, char* simulStart, bool doTraining, bool doRun);
EXPORT bool Ora_LogSaveRun(tDbg* dbg, tDBConnection* db, int pid, int tid, int barCnt, int featuresCnt, numtype* prediction, numtype* actual);
EXPORT bool Ora_LogLoadW(tDbg* dbg, tDBConnection* db, int pid, int tid, int epoch, int Wcnt, numtype* oW);

#ifdef __cplusplus
#undef EXPORT
#define EXPORT __declspec(dllexport)
#endif

