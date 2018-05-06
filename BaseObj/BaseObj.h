#pragma once

#include "../CommonMacros.h"
#include "../Debugger/Debugger.h"

#define OBJ_MAX_CHILDREN 64
#define OBJ_NAME_MAXLEN	128

struct sBaseObj {

	char objName[OBJ_NAME_MAXLEN];
	sBaseObj* objParent;
	int stackLevel;
	int childrenCnt=0;
	sBaseObj* child[OBJ_MAX_CHILDREN];

	sDebugger* dbg;

	EXPORT sBaseObj(char* objName_, sBaseObj* objParent_, sDebuggerParms* dbgparms_);

	EXPORT ~sBaseObj();

	void cleanup();

};

struct sDio : sBaseObj {
	int prop1;
	int prop2;

	sDio* childDio1;
	sDio* childDio2;
	sDio* childDio3;

	sDio(char* objName_, sBaseObj* objParent_, int prop1_, int prop2_, int children_=0, bool fail_=false, sDebuggerParms* dbgparms_=nullptr);

	void method(bool fail_);
};

//-- sRoot should be in the client (???)
/*
struct sRoot : sBaseObj {

	EXPORT sRoot(sDebuggerParms* rootdbgparms_=nullptr);

};
*/