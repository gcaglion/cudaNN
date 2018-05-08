#pragma once
#include "../CommonEnv.h"
#include "../DebuggerParms/DebuggerParms.h"
#include "s0macros.h"

typedef struct sdebuggerC {
	char* name;
	int stackLevel;
	sDebuggerParms* dbgparms;
	char dbgmsgmask[DBG_MSG_MAXLEN];
	char dbgmsg[DBG_MSG_MAXLEN];
	char stackmsg[DBG_STACK_MAXLEN];
} tdebuggerC;

struct s0 {
	char name[OBJ_NAME_MAXLEN];
	int stackLevel=0;
	s0* parent;
	int childrenCnt=0;
	s0* child[OBJ_MAX_CHILDREN];

	sDebuggerParms* dbgparms;
	char dbgmsgmask[DBG_MSG_MAXLEN]="";
	char dbgmsg[DBG_MSG_MAXLEN]="";
	char stackmsg[DBG_STACK_MAXLEN]="";

#ifdef __cplusplus__
	EXPORT s0(char* name_, s0* parent_, sDebuggerParms* dbgparms_);
	EXPORT ~s0();

	EXPORT void cleanup();
	void createDebugger(sDebuggerParms* dbgparms_);
	EXPORT void failmethod(int p);
	EXPORT void _fout(bool success);

	tdebuggerC* Cdbg;
#endif

};

//
