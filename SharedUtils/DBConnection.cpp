#include "DBConnection.h"

//=== sDBConnection
sDBConnection::sDBConnection(char* username, char* password, char* connstring, tDebugger* dbg_) {
	dbg=(dbg_==nullptr) ? (new tDebugger(new tFileInfo("DBConnection.err"))) : dbg_;
	strcpy_s(DBUser, DBUSER_MAXLEN, username);
	strcpy_s(DBPassword, DBPASSWORD_MAXLEN, password);
	strcpy_s(DBConnString, DBCONNSTRING_MAXLEN, connstring);
	DBCtx=NULL;
}
sDBConnection::sDBConnection(tParmsSource* iniParms, tDebugger* dbg_) {
	dbg=(dbg_==nullptr) ? (new tDebugger(new tFileInfo("DBConnection.err"))) : dbg_;
	iniParms->get(DBUser, "DBConnection.DBUser");
	iniParms->get(DBPassword, "DBConnection.DBPassword");
	iniParms->get(DBConnString, "DBConnection.DBConnString");
	DBCtx=NULL;
}
sDBConnection::sDBConnection() {}
sDBConnection::~sDBConnection() { delete dbg; }

