#pragma once

// 	for(int t=0; t<stackLevel; t++) printf("\t"); \

#define err(mask, ...) { \
	printf(mask, __VA_ARGS__); printf("\n"); \
	fprintf(dbg->outfile->handle, mask, __VA_ARGS__); \
	fprintf(dbg->outfile->handle, "\n"); \
}
#define info_(mask, ...){ if(dbg->verbose) err(mask, __VA_ARGS__); }

//-- create new child object
#define newC(child, newCmd){ \
	try{ \
		info_("Trying: %s = %s ...", #child, #newCmd); \
		child = newCmd; \
		subObj[subObjCnt]=child; \
		subObjCnt++; \
		info_("%s = %s completed successfully.", #child, #newCmd); \
	} catch (std::exception exc) { \
		sprintf_s(dbg->errmsg, DBG_ERRMSG_SIZE, "%s()->%s() failed to create %s . Exception=%s", objName, __func__, #child, exc.what()); \
		throw std::exception(dbg->errmsg); \
	} \
}

//-- method call
#define callM(cmd){ \
	try{ \
		info_("Trying: %s ...", #cmd); \
		cmd; \
		info_("%s completed successfully.", #cmd); \
	} catch (std::exception exc) { \
		sprintf_s(dbg->errmsg, DBG_ERRMSG_SIZE, "%s()->%s() failed. Exception=%s", objName, __func__, exc.what()); \
		throw std::exception(dbg->errmsg); \
	} \
}

#define start(...){ \
	sprintf_s(errtmp, DBG_ERRMSG_SIZE, __VA_ARGS__); \
	info_("%s->%s() called. %s", objName, __func__, errtmp); \
}
//-- constructor / method success
#define success(...){ \
	info_("%s->%s() successful.", objName, __func__); \
}

#define mainSuccess()	{ printf("main() success.\n"); system("pause"); return -1; } 
#define mainFail(...)	{ printf(__VA_ARGS__); system("pause"); return  0; }

#define fail(...){ \
	sprintf_s(errtmp, DBG_ERRMSG_SIZE, __VA_ARGS__); \
	sprintf_s(dbg->errmsg, DBG_ERRMSG_SIZE, "%s->%s() failed. Reason: %s", objName, __func__, errtmp); \
	err("stackLevel=%d ; %s->%s() failed. Reason: %s", stackLevel, objName, __func__, dbg->errmsg); \
	throw std::exception(dbg->errmsg); \
}

//	if(stackLevel>0) throw std::exception(dbg->errmsg); \
