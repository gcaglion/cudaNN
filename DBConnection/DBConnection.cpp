#include "DBConnection.h"

//=== sDBConnection
sDBConnection::sDBConnection(char* objName_, sBaseObj* objParent_, char* username, char* password, char* connstring, sDebuggerParms* dbgparms_) : sBaseObj(objName_, objParent_, dbgparms_) {
	strcpy_s(DBUser, DBUSER_MAXLEN, username);
	strcpy_s(DBPassword, DBPASSWORD_MAXLEN, password);
	strcpy_s(DBConnString, DBCONNSTRING_MAXLEN, connstring);
	DBCtx=NULL;
}
sDBConnection::sDBConnection(char* objName_, sBaseObj* objParent_, tParmsSource* parms, char* parmKey, sDebuggerParms* dbgparms_) : sBaseObj(objName_, objParent_, dbgparms_) {
	DBUser[DBUSER_MAXLEN]='\0';
	DBPassword[DBPASSWORD_MAXLEN]='\0';
	DBConnString[DBCONNSTRING_MAXLEN]='\0';

	safecall(parms->setKey(parmKey));
	parms->get(DBUser, "DBUser");
	parms->get(DBPassword, "DBPassword");
	parms->get(DBConnString, "DBConnString");
	DBCtx=NULL;
}

