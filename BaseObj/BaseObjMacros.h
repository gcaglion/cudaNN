#pragma once

#define start(mask, ...) { \
	msgbld("%s->%s(%s) starting.", this, __func__, __VA_ARGS__); \
}

//-- this is only needed so we can pass parentdbg's object address to child's constructor
#define spawn(objname, objtype, ...){ \
	objname = new objtype(#objname, this, __VA_ARGS__); \
	child[childrenCnt]=objname; \
	childHasDbg[childrenCnt]=(typeid(objname)==typeid(this)); \
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
#define info(mask, ...) { if(dbg->parms->verbose) err_(mask, __VA_ARGS__); }
#define err(mask, ...) { \
	err_(mask, __VA_ARGS__); \
	if (dbg->parms->pauseOnError) { printf("Press any key..."); getchar(); } \
}
#define err_(mask, ...) { \
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