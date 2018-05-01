#pragma once

//-- these are used by main()
#define mainSuccess()	{ printf("main() success.\n"); system("pause"); return -1; } 
#define mainFail(...)	{ printf(__VA_ARGS__); system("pause"); return  0; }

//-- basic output
#define err(mask, ...) { \
	printf(mask, __VA_ARGS__); printf("\n"); \
	fprintf(dbg->outfile->handle, mask, __VA_ARGS__); \
	fprintf(dbg->outfile->handle, "\n"); \
}
#define info(mask, ...){ if(dbg->verbose) err(mask, __VA_ARGS__); }



//-- this is called at the beginning of method or constructor
#define start(...){ \
	sprintf_s(dbg->stackmsg, DBG_ERRMSG_SIZE, __VA_ARGS__); \
	info("%s->%s() called. %s", objName, __func__, dbg->stackmsg); \
}

//-- create new child object
#define newC(child, newCmd){ \
	try{ \
		info("Trying: %s = %s ...", #child, #newCmd); \
		child = new newCmd; \
		child->setParent(this); \
		subObj[subObjCnt]=child; \
		subObjCnt++; \
		info("%s = %s completed successfully.", #child, #newCmd); \
	} catch (std::exception exc) { \
		sprintf_s(dbg->stackmsg, DBG_ERRMSG_SIZE, "%s\n\t%s()->%s() failed to create %s . Exception=%s", dbg->stackmsg, objName, __func__, #child, exc.what()); \
		cleanup(dbg->stackmsg); \
		throw std::exception(dbg->stackmsg); \
	} \
}

//-- constructor / method success
#define success(...){ \
	info("%s->%s() successful.", objName, __func__); \
}

//-- constructor failure
#define failC(mask, ...){ \
	err(mask, __VA_ARGS__); \
	sprintf_s(dbg->stackmsg, DBG_ERRMSG_SIZE, "%s\n\t%s()->%s() failed.", dbg->stackmsg, objName, __func__); \
	cleanup(dbg->stackmsg); \
	throw std::exception(dbg->stackmsg); \
}

