#include "DBConnection.h"

//=== sDBConnection
sDBConnection::sDBConnection(char* username, char* password, char* connstring, tDebugger* dbg_) {
	dbg=(dbg_==nullptr) ? (new tDebugger(new tFileInfo("DBConnection.err"))) : dbg_;
	strcpy_s(DBUser, DBUSER_MAXLEN, username);
	strcpy_s(DBPassword, DBPASSWORD_MAXLEN, password);
	strcpy_s(DBConnString, DBCONNSTRING_MAXLEN, connstring);
	DBCtx=NULL;
}
sDBConnection::sDBConnection(tParmsSource* iniParms, char* parmKey, tDebugger* dbg_) {
	dbg=(dbg_==nullptr) ? (new tDebugger(new tFileInfo("DBConnection.err"))) : dbg_;
	
	safeCallEB(iniParms->setKey(parmKey));
	iniParms->get(DBUser, "DBUser");
	iniParms->get(DBPassword, "DBPassword");
	iniParms->get(DBConnString, "DBConnString");
	DBCtx=NULL;
}
sDBConnection::sDBConnection() {}
sDBConnection::~sDBConnection() { delete dbg; }

