#pragma once
#include "../CommonEnv.h"
#include "../s0/s0.h"
#include "../FileInfo/FileInfo.h"
#include "../DebuggerParms/DebuggerParms.h"

typedef struct sDebugger : s0 {

	tFileInfo* outFile;
	char msg[DBG_MSG_MAXLEN]="";
	char stackmsg[DBG_STACK_MAXLEN];


#ifdef __cplusplus
	EXPORT sDebugger(char* objName_, s0* objParent_, sDebuggerParms* dbgparms_);
	EXPORT ~sDebugger();
#endif

} tDebugger;

