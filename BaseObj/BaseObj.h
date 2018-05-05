#pragma once

#include <typeinfo.h>
#include "BaseObjMacros.h"
#include "../Debugger/Debugger.h"

#define BASEOBJ_MAX_CHILDREN 64
struct sBaseObj {

	char objName[64];
	sBaseObj* objParent;
	int stackLevel;
	int childrenCnt=0;

	bool childHasDbg[BASEOBJ_MAX_CHILDREN];
	void* child[BASEOBJ_MAX_CHILDREN];

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

struct sRoot : sBaseObj {

	EXPORT sRoot(sDebuggerParms* rootdbgparms_=nullptr);

};
