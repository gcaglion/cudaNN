#pragma once
#include "../CommonEnv.h"
#include "../DebuggerParms/DebuggerParms.h"
#include "s0macros.h"

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

	s0(char* name_, s0* parent_, sDebuggerParms* dbgparms_) {
		strcpy_s(name, OBJ_NAME_MAXLEN, name_);
		parent=parent_;
		stackLevel=(parent==nullptr) ? 0 : parent->stackLevel+1;
		createDebugger(dbgparms_);
	}
	~s0() {
		cleanup();
	}

	void cleanup() {
		for (int c=0; c<childrenCnt; c++) {
			delete child[c];
		}
		delete dbgparms;
	}

	void createDebugger(sDebuggerParms* dbgparms_){
		dbgparms=(dbgparms_==nullptr) ? (new sDebuggerParms(name)) : dbgparms_;
		if (dbgparms->dest!=DBG_DEST_SCREEN) {
			info("Trying to create debugger outFile (%s)...", dbgparms->outFileFullName);
			try {
				dbgparms->outFile=new tFileInfo(dbgparms->outFileFullName, FILE_MODE_WRITE);
				info("SUCCESS in create debugger outFile (%s).", dbgparms->outFileFullName);
			}
			catch (std::exception exc) {
				fail("FAILURE in create debugger outFile (%s). Exception: %s", dbgparms->outFileFullName, exc.what());
			}
		}
	}

	void failmethod(int p) {
		if(p<0) throw( std::exception("failed method DioPorco."));
	}

	void _fout(bool success) {
		for (int t=0; t<stackLevel; t++) sprintf_s(dbgmsg, DBG_MSG_MAXLEN,"\t%s", dbgmsg); 
		strcat_s(dbgmsg, DBG_MSG_MAXLEN, "\n"); 
		strcat_s(stackmsg, DBG_STACK_MAXLEN, dbgmsg); 
		if (stackLevel>0) sprintf_s(parent->stackmsg, DBG_STACK_MAXLEN, "%st%s", parent->stackmsg, dbgmsg); 
		printf("%s", dbgmsg); 
		if (dbgparms->outFile!=nullptr) fprintf(dbgparms->outFile->handle, "%s", dbgmsg); 
		if (!success && dbgparms->pauseOnError) { printf("Press any key..."); getchar(); } 
	}

};
