#include "DBConnection.h"

//=== sDBConnection
sDBConnection::sDBConnection(char* username, char* password, char* connstring, tDebugger* dbg_) : sBaseObj("DBConnection", dbg_) {
	strcpy_s(DBUser, DBUSER_MAXLEN, username);
	strcpy_s(DBPassword, DBPASSWORD_MAXLEN, password);
	strcpy_s(DBConnString, DBCONNSTRING_MAXLEN, connstring);
	DBCtx=NULL;
}
sDBConnection::sDBConnection(tParmsSource* parms, char* parmKey, tDebugger* dbg_) : sBaseObj("DBConnection", dbg_) {
	DBUser[DBUSER_MAXLEN]='\0';
	DBPassword[DBPASSWORD_MAXLEN]='\0';
	DBConnString[DBCONNSTRING_MAXLEN]='\0';

	safeCall(parms->setKey(parmKey));
	parms->get(DBUser, "DBUser");
	parms->get(DBPassword, "DBPassword");
	parms->get(DBConnString, "DBConnString");
	DBCtx=NULL;
}
sDBConnection::~sDBConnection() {
}

