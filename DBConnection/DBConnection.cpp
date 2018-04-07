#include "DBConnection.h"

//=== sDBConnection
sDBConnection::sDBConnection(char* username, char* password, char* connstring, tDebugger* dbg_) {
	dbg=(dbg_==nullptr) ? (new tDebugger("DBConnection.err")) : dbg_;
	strcpy_s(DBUser, DBUSER_MAXLEN, username);
	strcpy_s(DBPassword, DBPASSWORD_MAXLEN, password);
	strcpy_s(DBConnString, DBCONNSTRING_MAXLEN, connstring);
	DBCtx=NULL;
}
sDBConnection::sDBConnection(tParmsSource* parms, char* parmKey, tDebugger* dbg_) {
	dbg=(dbg_==nullptr) ? (new tDebugger("DBConnection.err")) : dbg_;
	DBUser[DBUSER_MAXLEN]='\0';
	DBPassword[DBPASSWORD_MAXLEN]='\0';
	DBConnString[DBCONNSTRING_MAXLEN]='\0';

	safeCallEB(parms->setKey(parmKey));
	parms->get(DBUser, "DBUser");
	parms->get(DBPassword, "DBPassword");
	parms->get(DBConnString, "DBConnString");
	DBCtx=NULL;
}
sDBConnection::sDBConnection() {}
sDBConnection::~sDBConnection() {
	delete dbg;
}

