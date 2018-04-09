#pragma once

#include "../CommonEnv.h"
#include "Core_enums.h"
#include "../Utils/Utils.h"
#include "../ParamMgr/ParamMgr.h"

#define CORE_MAX_DESC_LEN	128
#define CORE_MAX_PARENTS	32

#define coreCalled

typedef struct sCoreLayout : sBaseObj {
	int Id;
	char* desc;
	int layer=0;
	int type;
	int parentsCnt;
	int* parentId;
	char** parentDesc;
	int* parentConnType;

	int inputCnt;
	int outputCnt;

	void sCoreLayout_common(tDebugger* dbg_, int Id_);
	EXPORT sCoreLayout(tParmsSource* parms, int Id_, int outputCnt_, tDebugger* dbg_=nullptr);
	EXPORT ~sCoreLayout();
	EXPORT void setLayer();

} tCoreLayout;

typedef struct sCore {

	tCoreLayout* layout;

	EXPORT sCore();
	EXPORT ~sCore();



} tCore;