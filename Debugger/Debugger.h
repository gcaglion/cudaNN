#pragma once
#include "../CommonEnv.h"
#include "../FileInfo/FileInfo.h"
#include "Debugger_enums.h"
#include "DebuggerParms.h"

typedef struct sDebugger {

	tDebuggerParms* parms;
	tFileInfo* outFile;

	char msg[DBG_MSG_MAXLEN]
#ifdef __cplusplus
		=""
#endif
		;
	char stackmsg[DBG_STACK_MAXLEN]
#ifdef __cplusplus
		=""
#endif
		;

#ifdef __cplusplus
	EXPORT sDebugger(sDebuggerParms* dbgparms_=nullptr);
	EXPORT ~sDebugger();
#endif

} tDebugger;
