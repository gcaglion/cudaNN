#pragma once

#define setmsg(mask, ...){ \
		sprintf_s(dbgmsgmask, DBG_MSG_MAXLEN, "%s(%p)->%s() %s:", name, this, __func__, mask); \
		sprintf_s(dbgmsg, DBG_MSG_MAXLEN, dbgmsgmask, __VA_ARGS__); \
}
#define info(mask, ...)  { \
	if(dbgparms->verbose) { \
		setmsg(mask, __VA_ARGS__); \
		_fout(true); \
	} \
}

#define err(mask, ...) { \
	setmsg(mask, __VA_ARGS__); \
	_fout(false); \
}

#define fail(mask, ...) { \
	err(mask, __VA_ARGS__); \
	cleanup(); \
	throw(std::exception(dbgmsg)); \
}

#define spawn(objname, objtype, ...){ \
	objname = new objtype(#objname, this, __VA_ARGS__); \
	child[childrenCnt]=objname; \
	childrenCnt++; \
}

#define safecall(...) { \
	info("Calling: %s ...", #__VA_ARGS__); \
	try{ \
		__VA_ARGS__; \
		info("SUCCESS: %s", #__VA_ARGS__); \
	} catch(std::exception exc){ \
		fail("FAILURE: %s ; Exception: %s", #__VA_ARGS__, exc.what()); \
	} \
}

#define safespawn(objname, objtype, ...){ \
	safecall(spawn(objname, objtype, __VA_ARGS__)); \
}