#pragma once

#include "../CommonEnv.h"
#include "Core_enums.h"
#include "../Utils/Utils.h"
#include "../ParamMgr/ParamMgr.h"
#include "../Data/Data.h"

#define CORE_MAX_DESC_LEN	128
#define CORE_MAX_PARENTS	32

#define coreCalled

typedef struct sCoreLayout : sBaseObj {
	int Id;
	char* desc;
	int layer=-1;
	int type;
	int parentsCnt;
	int* parentId;
	char** parentDesc;
	int* parentConnType;

	tDataShape* shape;

	void sCoreLayout_common(tDebugger* dbg_, int Id_);
	EXPORT sCoreLayout(tParmsSource* parms, int Id_, tDataShape* shape_, tDebugger* dbg_=nullptr);
	EXPORT ~sCoreLayout();

} tCoreLayout;

typedef struct sCore {

	int kaz;
	tCoreLayout* layout;

	EXPORT sCore();
	EXPORT sCore(tParmsSource* parms, tCoreLayout* layout_);
	EXPORT ~sCore();



} tCore;