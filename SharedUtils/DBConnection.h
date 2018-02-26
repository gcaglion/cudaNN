// Database Connection - generic (can be used for both data retrieval and logging)
#pragma once
#include "../CommonEnv.h"
#include "DebugInfo.h"

typedef struct sDBConnection{
	char DBUser[30];
	char DBPassword[30];
	char DBConnString[30];
	void* DBCtx;
	tDbg* dbg;
#ifdef __cplusplus
	EXPORT sDBConnection(char* username, char* password, char* connstring, tDbg* dbg_=nullptr);
	EXPORT sDBConnection();
	EXPORT ~sDBConnection();
#endif
} tDBConnection;
