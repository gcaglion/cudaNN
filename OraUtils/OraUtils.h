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
#include "../s0/s0.h"
#include "../DBConnection/DBConnection.h"
#include <math.h>

//=== DB common functions
EXPORT bool OraConnect(tDebuggerC* dbg, tDBConnection* DBConnInfo);
EXPORT void OraDisconnect(tDBConnection* DBConnInfo, int Commit);
EXPORT void OraCommit(tDBConnection* DBConnInfo);

//=== Retrieval functions
EXPORT bool Ora_GetFlatOHLCV(tDebuggerC* debugger, tDBConnection* db, char* pSymbol, char* pTF, char* pDate0, int pRecCount, char** oBarTime, float* oBarData, char* oBaseTime, float* oBaseBar);

//=== Logging functions
EXPORT bool Ora_LogSaveMSE(tDebuggerC* debugger, tDBConnection* db, int pid, int tid, int mseCnt, float* mseT, float* mseV);
EXPORT bool Ora_LogSaveW(tDebuggerC* debugger, tDBConnection* db, int pid, int tid, int epoch, int Wcnt, numtype* W);
EXPORT bool Ora_LogSaveClient(tDebuggerC* debugger, tDBConnection* db, int pid, char* clientName, DWORD startTime, DWORD duration, int simulLen, char* simulStart, bool doTrain, bool doTrainRun, bool doTestRun);
EXPORT bool Ora_LogSaveRun(tDebuggerC* debugger, tDBConnection* db, int pid, int tid, int setid, int npid, int ntid, int barCnt, int featuresCnt, int* feature, numtype* prediction, numtype* actual);
EXPORT bool Ora_LogLoadW(tDebuggerC* debugger, tDBConnection* db, int pid, int tid, int epoch, int Wcnt, numtype* oW);

#ifdef __cplusplus
#undef EXPORT
#define EXPORT __declspec(dllexport)
#endif
