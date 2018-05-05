#pragma once


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
