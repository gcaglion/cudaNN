#pragma once
#include "../CommonEnv.h"
#include "../SharedUtils/ParamMgr_limits.h"

//-- this is used by both Engine.h and Core.h

#define MAX_ENGINE_LAYERS	16

typedef struct sEngineLayout {
	int layersCnt=0;
	int* layerCoresCnt;
	int coresCnt;
	int* coreLayer;
	int* coreType;
	int*  coreParentsCnt;
	int** coreParent;
	int** coreParentConnType;
	char*** coreParentDesc;
	int* coreInputCnt;
	int* coreOutputCnt;

	EXPORT sEngineLayout(int coresCnt_);
	EXPORT ~sEngineLayout();

	EXPORT int getCoreLayer(int c);

} tEngineLayout;
