#pragma once
typedef void * sql_context;

#include "../CommonEnv.h"

#include "../fxdata.h"
#include "../MyDebug/mydebug.h"

#ifdef __cplusplus
#undef EXPORT
#define EXPORT extern "C" __declspec(dllexport)
#endif

//=== DB common functions
EXPORT int  OraConnect(tDebugInfo* DebugInfo, tDBConnection* DBConnInfo);
EXPORT void OraDisconnect(sql_context pCtx, int Commit);
EXPORT void OraCommit(void* pCtx);
//===

EXPORT int GetFlatBarsFromQuery(tDebugInfo* DebugParms, sql_context pCtx, char* pSQL, int pRecCount, float* oBarData, float* oBaseBar);
EXPORT int GetBarsFromQuery(tDebugInfo* DebugParms, sql_context pCtx, char* pSQL, int pRecCount, int pSkipFirstN, tBar* oBar);
EXPORT int GetCharPFromQuery(tDebugInfo* DebugParms, sql_context pCtx, char* pSQL, char* oRet);
EXPORT int GetStringArrayFromQuery(tDebugInfo* DebugParms, sql_context pCtx, char* pSQL, int ArrLen, char** oRet);

#ifdef __cplusplus
#undef EXPORT
#define EXPORT __declspec(dllexport)
#endif

