#include "DBConnection.h"

//=== sDBConnection
sDBConnection::sDBConnection(char* username, char* password, char* connstring, tDbg* dbg_) {
	dbg=(dbg_==nullptr) ? (new tDbg(DBG_LEVEL_ERR, DBG_DEST_FILE, new tFileInfo("DBConnection.err"))) : dbg_;
	strcpy_s(DBUser, DBUSER_MAXLEN, username);
	strcpy_s(DBPassword, DBPASSWORD_MAXLEN, password);
	strcpy_s(DBConnString, DBCONNSTRING_MAXLEN, connstring);
	DBCtx=NULL;
}
sDBConnection::sDBConnection(tParamMgr* iniParms, tDbg* dbg_) {
	dbg=(dbg_==nullptr) ? (new tDbg(DBG_LEVEL_ERR, DBG_DEST_FILE, new tFileInfo("DBConnection.err"))) : dbg_;
	iniParms->getx(DBUser, "DBUser");
	iniParms->getx(DBPassword, "DBPassword");
	iniParms->getx(DBConnString, "DBConnString");
	DBCtx=NULL;
}
sDBConnection::sDBConnection() {}
sDBConnection::~sDBConnection() { delete dbg; }

