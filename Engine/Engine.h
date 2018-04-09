#pragma once
#include "../CommonEnv.h"
#include "../Debugger/Debugger.h"
#include "../ParamMgr/ParamMgr.h"
#include "../Data/Data.h"
#include "../Core/Core.h"
#include "Engine_enums.h"

#include "../cuNN/cuNN.h"

#define MAX_ENGINE_LAYERS	8
#define MAX_ENGINE_CORES	32

/*
typedef struct sEngineLayout {
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

} tEngineLayout;
*/
typedef struct sEngine : public sBaseObj {

	int type;
	int coresCnt;
	int layersCnt=0;
	int* layerCoresCnt;

	tDataShape* shape;

	tCoreLayout** coreLayout;
	tCore** core;

	EXPORT void sEngine::sEngine_common(tParmsSource* parms, tDataShape* shape_, tDebugger* dbg_);
	EXPORT sEngine(tParmsSource* parms, char* parmKey, tDataShape* shape_, tDebugger* dbg_=nullptr);
	EXPORT ~sEngine();

	EXPORT void setCoreLayer(tCoreLayout* c);
	EXPORT void train(tDataSet* trainDS);
	EXPORT void addCore(tCoreLayout* coreLayout_);

	int getMaxLayer(int coreIdCnt, int* coreIdList_);

} tEngine;