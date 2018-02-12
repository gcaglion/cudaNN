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
EXPORT int  OraConnect(tDebugInfo* DebugInfo, tDBConnection* DBConnInfo);
EXPORT void OraDisconnect(tDBConnection* DBConnInfo, int Commit);
EXPORT void OraCommit(tDBConnection* DBConnInfo);

//=== Retrieval functions
EXPORT int Ora_GetFlatOHLCV(tDebugInfo* DebugParms, tDBConnection* db, char* pSymbol, char* pTF, char* pDate0, int pRecCount, char** oBarTime, float* oBarData, char* oBaseTime, float* oBaseBar);

/*EXPORT int GetFlatBarsFromQuery(tDebugInfo* DebugParms, sql_context pCtx, char* pSQL, int pRecCount, float* oBarData, float* oBaseBar);
EXPORT int GetBarsFromQuery(tDebugInfo* DebugParms, sql_context pCtx, char* pSQL, int pRecCount, int pSkipFirstN, tBar* oBar);
EXPORT int GetCharPFromQuery(tDebugInfo* DebugParms, sql_context pCtx, char* pSQL, char* oRet);
EXPORT int GetStringArrayFromQuery(tDebugInfo* DebugParms, sql_context pCtx, char* pSQL, int ArrLen, char** oRet);
*/

//=== Logging functions
EXPORT int Ora_LogSaveMSE(tDebugInfo* DebugParms, tDBConnection* db, int pid, int tid, int mseCnt, float* mseT, float* mseV);
EXPORT int Ora_LogSaveW(tDebugInfo* DebugParms, tDBConnection* db, int pid, int tid, int epoch, int Wcnt, numtype* W);
EXPORT int Ora_LogSaveClient(tDebugInfo* DebugParms, tDBConnection* db, int pid, char* clientName, DWORD startTime, DWORD duration, int simulLen, char* simulStart, int doTraining, int doRun);
EXPORT int Ora_LogSaveRun(tDebugInfo* DebugParms, tDBConnection* db, int pid, int tid, int barCnt, int featuresCnt, numtype* prediction, numtype* actual);
EXPORT int Ora_LogLoadW(tDebugInfo* DebugParms, tDBConnection* db, int pid, int tid, int epoch, int Wcnt, numtype* oW);

#ifdef __cplusplus
#undef EXPORT
#define EXPORT __declspec(dllexport)
#endif

