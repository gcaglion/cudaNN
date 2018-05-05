#pragma once

#include "../CommonEnv.h"
#include "Core_enums.h"
#include "../Utils/Utils.h"
#include "../ParamMgr/ParamMgr.h"
#include "../Data/Data.h"
#ifdef USE_GPU
#include <cuda_runtime.h>
#endif

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

	void sCoreLayout_common(sDebuggerParms* dbgparms_, int Id_);
	EXPORT sCoreLayout(char* objName_, sBaseObj* objParent_, tParmsSource* parms, int Id_, tDataShape* shape_, sDebuggerParms* dbgparms_=nullptr);
	//-- TO DO !!! EXPORT sCoreLayout(char* objName_, sBaseObj* objParent_, tParmsSource* parms, int Id_, tDataShape* shape_, sDebuggerParms* dbgparms_=nullptr);

	EXPORT ~sCoreLayout();

} tCoreLayout;

typedef struct sCore : sBaseObj {

	int kaz;
	tCoreLayout* layout;

	EXPORT sCore(char* objName_, sBaseObj* objParent_, tParmsSource* parms, tCoreLayout* layout_, sDebuggerParms* dbgparms_=nullptr);
	EXPORT sCore(char* objName_, sBaseObj* objParent_, tDataShape* baseShape_, sDebuggerParms* dbgparms_=nullptr);

} tCore;