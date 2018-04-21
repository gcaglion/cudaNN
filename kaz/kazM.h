#pragma once

// 	for(int t=0; t<stackLevel; t++) printf("\t"); \

#define classErr(...){ \
	sprintf_s(errtmp, DBG_ERRMSG_SIZE, __VA_ARGS__); \
	sprintf_s(errmsg, DBG_ERRMSG_SIZE, "%s->%s() failed. Reason: %s", objName, __func__, errtmp); \
	err("%s->%s() failed. Reason: %s", objName, __func__, errmsg); \
}

#define err(mask, ...){ \
	printf(mask, __VA_ARGS__); printf("\n"); \
	fprintf(dbg->outfile->handle, mask, __VA_ARGS__); \
	fprintf(dbg->outfile->handle, "\n"); \
}
#define info(mask, ...){ if(dbg->verbose) err(mask, __VA_ARGS__); }

#define spawn(child, newCmd){ \
	try{ \
		info("Trying: %s = %s ...", #child, #newCmd); \
		child = newCmd; \
		subObj[subObjCnt]=child; \
		subObjCnt++; \
		info("%s = %s completed successfully.", #child, #newCmd); \
	} catch (std::exception exc) { \
		sprintf_s(errmsg, DBG_ERRMSG_SIZE, "%s()->%s() failed to create %s . Exception=%s", objName, __func__, #child, exc.what()); \
		cleanup(errmsg); \
		throw std::exception(errmsg); \
	} \
}

#define method(cmd){ \
	try{ \
		info("Trying: %s ...", #cmd); \
		cmd; \
		info("%s completed successfully.", #cmd); \
	} catch (std::exception exc) { \
		sprintf_s(errmsg, DBG_ERRMSG_SIZE, "%s()->%s() failed. Exception=%s", objName, __func__, exc.what()); \
		cleanup(errmsg); \
		throw std::exception(errmsg); \
	} \
}

#define successM() { \
	info("%s->%s() successful.", objName, __func__); \
}

#define failM(...){ \
	classErr(__VA_ARGS__); \
	cleanup(errmsg); \
	throw std::exception(errmsg); \
}

#define mainSuccess()	{ printf("main() success.\n"); system("pause"); return -1; } 
#define mainFail(...)	{ printf(__VA_ARGS__); 	delete dbg; system("pause"); return  0; }

