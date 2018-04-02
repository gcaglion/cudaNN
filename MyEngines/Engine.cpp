#include "Engine.h"

sEngine::sEngine(int type_, tDataShape* shape_, tDebugger* dbg_){
	dbg=(dbg_==nullptr) ? (new tDebugger(new tFileInfo("Engine.err"))) : dbg_;
	type=type_; shape=shape_;

}
sEngine::sEngine(tParmsSource* parms, char* parmKey, tDataShape* shape_, tDebugger* dbg_){
	dbg=(dbg_==nullptr) ? (new tDebugger(new tFileInfo("Engine.err"))) : dbg_; //-- TO DO: How to handle specific <Debugger>/</Debugger> info??
	shape=shape_;
	int coresCnt_; 
	int c;
	char coreKey[XML_MAX_PARAM_NAME_LEN];

	safeCallEB(parms->setKey(parmKey));
	parms->get(&type, "Type");

	switch (type) {
	case ENGINE_CUSTOM:
		safeCallEB(parms->setKey("Custom"));
		safeCallEB(parms->backupKey());

		//-- 0. temporary coresCnt
		safeCallEE(parms->get(&coresCnt_, "CoresCount"));

		//-- 1. create Engine layout object
		safeCallEE(layout=new tEngineLayout(coresCnt_));
		//-- 1.1. set layout, and outputCnt for each Core
		for(c=0; c<layout->coresCnt; c++) {
			sprintf_s(coreKey, XML_MAX_PARAM_NAME_LEN, "Core%d", c);
			safeCallEB(parms->setKey(coreKey));
			sprintf_s(layout->coreDesc[c], XML_MAX_PARAM_NAME_LEN, coreKey);
			safeCallEE(parms->get(&layout->coreType[c], "Type"));
			safeCallEE(parms->get(layout->coreParentDesc[c], "Parents"));
			safeCallEE(parms->get(&layout->coreParentConnType[c], "ParentsConnType", &layout->coreParentsCnt[c]));
			//-- outputCnt is assumed to be the same across all Cores, and dependent on DataShape
			layout->coreOutputCnt[c]=shape->predictionLen*shape->featuresCnt;

			safeCallEB(parms->restoreKey());
		}
		//-- 1.2. determine Layer for each Core, and cores count for each layer
		for (c=0; c<layout->coresCnt; c++) {
			layout->coreLayer[c]=layout->getCoreLayer(c);
			layout->layerCoresCnt[layout->coreLayer[c]]++;
		}
		//-- 1.3. determine layersCnt, and InputCnt for each Core
		for(int l=0; l<MAX_ENGINE_LAYERS; l++) {
			for(c=0; c<layout->layerCoresCnt[l]; c++) {
				if(l==0) {
					layout->coreInputCnt[c]=shape->sampleLen*shape->featuresCnt;
				} else {
					layout->coreInputCnt[c]=layout->layerCoresCnt[l-1]*layout->coreOutputCnt[c];
				}				
			}
			if (c==0) break;
			layout->layersCnt++;
		}

		//-- 2. create CoreLayouts and Cores
		core=(tCore**)malloc(layout->coresCnt*sizeof(tCore*));
		coreLayout=(tCoreLayout**)malloc(layout->coresCnt*sizeof(tCoreLayout*));
		for (c=0; c<layout->coresCnt; c++) {
			safeCallEB(parms->restoreKey());
			sprintf_s(coreKey, XML_MAX_PARAM_NAME_LEN, "Core%d", c);
			coreLayout[c]=new tCoreLayout(layout, c);
			core[c]=new tCore(parms, coreLayout[c]);
		}

		//-- 
		break;
	case ENGINE_WNN:
		safeCallEB(parms->setKey("WNN"));
		//... get() ...
		break;
	case ENGINE_XIE:
		safeCallEB(parms->setKey("XIE"));
		layout->coresCnt=3;
		//... get() ...
		break;
	default:
		throwE("Invalid Engine Type: %d", 1, type);
		break;
	}
}
sEngine::~sEngine(){
	delete layout;
	delete dbg;
}

