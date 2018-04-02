#pragma once

#include "../CommonEnv.h"
#include "../SharedUtils/ParamMgr_limits.h"

#define MAX_ENGINE_LAYERS	16
#define MAX_CORE_PARENTS	64

typedef struct sEngineLayout {
	int layersCnt=0;
	int* layerCoresCnt;
	int coresCnt;

	int* coreLayer;
	int* coreType;
	char** coreDesc;
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

typedef struct sCoreLayout {
	int id;
	char* desc;
	int layer;
	int type;
	int parentsCnt;
	int* parentId;
	int* parentConnType;

	int inputCnt;
	int outputCnt;

	EXPORT sCoreLayout(int id_, char* desc_, int layer_, int type_, int inputCnt_, int outputCnt_, int parentsCnt_, int* parentId_, int* parentConnType_);
	EXPORT sCoreLayout(tEngineLayout* engineLayout, int coreId);
	EXPORT ~sCoreLayout();

} tCoreLayout;
