// Database Connection - generic (can be used for both data retrieval and logging)
#pragma once
typedef struct sDBConnection{
	char DBUser[30];
	char DBPassword[30];
	char DBConnString[30];
	void* DBCtx;
#ifdef __cplusplus
	sDBConnection(char* username, char* password, char* connstring) {
		strcpy_s(DBUser, 30, username);
		strcpy_s(DBPassword, 30, password);
		strcpy_s(DBConnString, 30, connstring);
		DBCtx=NULL;
	}
	sDBConnection(){}
#endif
} tDBConnection;
