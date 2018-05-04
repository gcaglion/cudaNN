#pragma once
#include "..\FileInfo\FileInfo.h"

#define DEFAULT_DBG_FPATH "C:/temp/logs"
#define DEFAULT_DBG_FNAME "Debugger"
#define DEFAULT_DBG_VERBOSITY false
#define DEFAULT_DBG_TIMING false
#define DEFAULT_DBG_PAUSERR true
#define DBG_MSG_MAXLEN 1024
#define DBG_STACK_MAXLEN 32768

#define start(mask, ...) { \
	msgbld("%s->%s(%s) starting.", this, __func__, __VA_ARGS__); \
}

//-- this is only needed so we can pass parentdbg's object address to child's constructor
#define spawn(objname, objtype, ...){ \
	objname = new objtype(#objname, this, __VA_ARGS__); \
	child[childrenCnt]=objname; \
	childrenCnt++; \
}
#define safespawn(objname, objtype, ...) \
	objtype* objname=nullptr; \
	info("%s->%s() Trying:  %s = new %s(%s)...", objName, __func__, #objname, #objtype, #__VA_ARGS__); \
	try { \
		spawn(objname, objtype, __VA_ARGS__); \
		info("%s->%s() Success: %s = new %s(%s).", objName, __func__, #objname, #objtype, #__VA_ARGS__); \
	} catch (std::exception exc) { \
		fail("%s->%s() Failure: %s = new %s(%s). Exception: %s", objName, __func__, #objname, #objtype, #__VA_ARGS__, exc.what()); \
	} 

#define safecall(...) { \
	info("%s->%s() Trying:  %s...", objName, __func__, #__VA_ARGS__); \
	try { \
		__VA_ARGS__; \
		info("%s->%s() Success: %s ", objName, __func__, #__VA_ARGS__); \
	} catch (std::exception exc) { \
		fail("%s->%s() Failure: %s . Exception: ---%s---", objName, __func__, #__VA_ARGS__, exc.what()); \
	} \
}

//-- info() , err(), fail() for sBaseObj object types
#define info(mask, ...) { if(dbg->parms->verbose) err(mask, __VA_ARGS__); }
#define err(mask, ...) { \
	for(int t=0; t<stackLevel; t++) dbg->msg[t]='\t'; \
	sprintf_s(&dbg->msg[stackLevel], DBG_MSG_MAXLEN, mask, __VA_ARGS__); strcat_s(dbg->msg, DBG_MSG_MAXLEN, "\n"); \
	strcat_s(dbg->stackmsg, DBG_STACK_MAXLEN, dbg->msg); \
	if(stackLevel>0) sprintf_s(objParent->dbg->stackmsg, DBG_STACK_MAXLEN, "%s\t%s", objParent->dbg->stackmsg, dbg->msg); \
	printf("%s", dbg->msg); \
	fprintf(dbg->outFile->handle, "%s", dbg->msg); \
}
#define fail(mask, ...) { \
	err(mask, __VA_ARGS__); \
	throw(std::exception(dbg->msg)); \
}
//-- info() , err(), fail() for non-sBaseObj object types
#define info_d(mask, ...) { if(parms->verbose) err_d(mask, __VA_ARGS__); }
#define err_d(mask, ...) { \
	sprintf_s(msg, DBG_MSG_MAXLEN, mask, __VA_ARGS__); \
	printf("%s\n", msg); \
}
#define fail_d(mask, ...) { \
	err_d(mask, __VA_ARGS__); \
	throw(std::exception(stackmsg)); \
}
