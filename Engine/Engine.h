#pragma once
#include "../CommonEnv.h"
#include "../ParamMgr/ParamMgr.h"
#include "../Data/Data.h"
#include "../Core/Core.h"
#include "Engine_enums.h"

#include "../cuNN/cuNN.h"

#define MAX_ENGINE_CORES	32

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

typedef struct sEngine : public sBaseObj {

	int type;
	int coresCnt;

	tCore** core;

	EXPORT sEngine(tParmsSource* parms, char* parmKey, tDataShape* shape_, tDebugger* dbg_=nullptr);
	EXPORT ~sEngine();

	EXPORT void train(tDataSet* trainDS);
	EXPORT void addCore(tCoreLayout* coreLayout_);

} tEngine;