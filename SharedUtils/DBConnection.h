// Database Connection - generic (can be used for both data retrieval and logging)
#pragma once
#include "../CommonEnv.h"
#include "Debugger.h"
#include "DBConnection_enums.h"
#ifdef __cplusplus
#include "ParamMgr.h"
#endif

typedef struct sDBConnection{
	char DBUser[DBUSER_MAXLEN];
	char DBPassword[DBPASSWORD_MAXLEN];
	char DBConnString[DBCONNSTRING_MAXLEN];
	void* DBCtx;
	tDebugger* dbg;
#ifdef __cplusplus
	EXPORT sDBConnection(char* username, char* password, char* connstring, tDebugger* dbg_=nullptr);
	EXPORT sDBConnection(tParmsSource* iniParms, tDebugger* dbg_=nullptr);
	EXPORT sDBConnection();
	EXPORT ~sDBConnection();
#endif

} tDBConnection;
