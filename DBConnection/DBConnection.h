// Database Connection - generic (can be used for both data retrieval and logging)
#pragma once
#include "../CommonEnv.h"
#include "DBConnection_enums.h"
#ifdef __cplusplus
#include "../s0/s0.h"
#include "../ParamMgr/ParamMgr.h"
#endif

//-- limits
#define DBUSER_MAXLEN 30
#define DBPASSWORD_MAXLEN 30
#define DBCONNSTRING_MAXLEN 30

typedef struct sDBConnection
#ifdef __cplusplus
	: public s0
#endif
{
	char DBUser[DBUSER_MAXLEN];
	char DBPassword[DBPASSWORD_MAXLEN];
	char DBConnString[DBCONNSTRING_MAXLEN];
	void* DBCtx;

#ifdef __cplusplus
	EXPORT sDBConnection(char* objName_, s0* objParent_, char* username, char* password, char* connstring, sDebuggerParms* dbgparms_=nullptr);
	EXPORT sDBConnection(char* objName_, s0* objParent_, tParmsSource* parms, char* parmKey, sDebuggerParms* dbgparms_=nullptr);
	EXPORT sDBConnection();
	EXPORT ~sDBConnection();
#endif

} tDBConnection;
