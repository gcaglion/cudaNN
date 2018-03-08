// Database Connection - generic (can be used for both data retrieval and logging)
#pragma once
#include "../CommonEnv.h"
#include "DebugInfo.h"
#include "ParamMgr.h"
#include "DBConnection_enums.h"

typedef struct sDBConnection{
	char DBUser[DBUSER_MAXLEN];
	char DBPassword[DBPASSWORD_MAXLEN];
	char DBConnString[DBCONNSTRING_MAXLEN];
	void* DBCtx;
	tDbg* dbg;
#ifdef __cplusplus
	EXPORT sDBConnection(char* username, char* password, char* connstring, tDbg* dbg_=nullptr);
	EXPORT sDBConnection(tParamMgr* iniParms, tDbg* dbg_=nullptr);
	EXPORT sDBConnection();
	EXPORT ~sDBConnection();
#endif

} tDBConnection;
