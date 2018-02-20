// Database Connection - generic (can be used for both data retrieval and logging)
#pragma once
#include "MyDebug/mydebug.h"

typedef struct sDBConnection{
	char DBUser[30];
	char DBPassword[30];
	char DBConnString[30];
	void* DBCtx;
	tDebugInfo* DebugParms;
#ifdef __cplusplus
	sDBConnection(char* username, char* password, char* connstring, tDebugInfo* DebugParms_=nullptr) {
		if (DebugParms_==nullptr) {
			DebugParms=new tDebugInfo(DBG_LEVEL_ERR, DBG_DEST_FILE, new tFileInfo("DBConnection.err"));
		} else {
			DebugParms=DebugParms_;
		}
		strcpy_s(DBUser, 30, username);
		strcpy_s(DBPassword, 30, password);
		strcpy_s(DBConnString, 30, connstring);
		DBCtx=NULL;
	}
	sDBConnection(){}
#endif
} tDBConnection;
