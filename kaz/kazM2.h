#pragma once

//-- these are used by main()
#define mainSuccess()	{ printf("main() success.\n"); system("pause"); return -1; } 
#define mainFail(...)	{ printf(__VA_ARGS__); system("pause"); return  0; }

//-- basic output

//-- builds dbg->msg from Var args
#define buildmsg(mask, ...){ \
	printf("-- dbg->msg BEFORE: %s ; mask: %s\n", dbg->msg, mask); \
	sprintf_s(dbg->msg, DBG_ERRMSG_SIZE, __VA_ARGS__); \
	printf("-- dbg->msg AFTER: %s\n",dbg->msg); \
}

#define err(mask, ...) { \
	buildmsg(mask, __VA_ARGS__); \
	dbg->errd(); \
}
//#define info(mask, ...){ if(dbg->verbose) err(mask, __VA_ARGS__); }

#define info(mask, ...){ \
	sprintf_s(dbg->msg, DBG_ERRMSG_SIZE, "%s->%s() called. %s message=", objName, __func__, dbg->msg); \
}


#define start(mask, ...){ \
	buildmsg(mask, __VA_ARGS__); \
	getchar(); \
	printf("mask=%s ; dbg->msc=%s\n", #mask, dbg->msg); \
	info("%s->%s() called. start message = %s", objName, __func__, dbg->msg); \
}

//-- create new child object
#define newC(cname, ctype, ...){ \
	try{ \
		info("Trying: %s = new %s(%s) ...", #cname, #ctype, #__VA_ARGS__); \
		cname = new ctype(this, #cname, __VA_ARGS__); \
		subObj[subObjCnt]=cname; \
		subObjCnt++; \
		info("%s = new %s(%s) completed successfully.", #cname, #ctype, #__VA_ARGS__); \
	} catch (std::exception exc) { \
		sprintf_s(dbg->stackmsg, DBG_ERRMSG_SIZE, "%s\n\t%s()->%s() failed to create %s . Exception=%s", dbg->stackmsg, objName, __func__, #cname, exc.what()); \
		cleanup(dbg->stackmsg); \
		throw std::exception(dbg->stackmsg); \
	} \
}

//-- constructor / method success

#define success(){ \
	info("%s->%s() successful.", objName, __func__); \
}

//-- constructor failure
#define failC(mask, ...){ \
	err(#mask, __VA_ARGS__); \
	throw std::exception(((sLbase*)parentObj)->dbg->stackmsg); \
}

#define newC2(cname, ctype, ...) { \
	err("Trying: %s = new %s(%s) ...", #cname, #ctype, #__VA_ARGS__); \
}


#define nakedNew(cname, ctype, ...) { \
	cname = new ctype(this, #cname, __VA_ARGS__); \
}
#define nakedNewFail(){ \
}

#define new2(cname, ctype, ...) { \
}

