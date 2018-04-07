#include "Engine.h"

//-- Engine stuff
sEngine::sEngine(tParmsSource* parms, char* parmKey, tDataShape* shape_, tDebugger* dbg_) {
	dbg=(dbg_==nullptr) ? (new tDebugger("Engine.err")) : dbg_;	//... handle specific debugger in xml ...

	core=(tCore**)malloc(MAX_ENGINE_CORES*sizeof(tCore*));

	safeCallEB(parms->setKey("Custom"));
	safeCallEB(parms->backupKey());

	//-- 0. temporary coresCnt
	safeCallEE(parms->get(&coresCnt, "CoresCount"));

	//-- 1.1. set layout, and outputCnt for each Core
	char coreKey[XML_MAX_PARAM_NAME_LEN];
	for (int c=0; c<coresCnt; c++) {
		sprintf_s(coreKey, XML_MAX_PARAM_NAME_LEN, "Core%d", c);
		safeCallEB(parms->setKey(coreKey));
		sprintf_s(coreDesc[c], XML_MAX_PARAM_NAME_LEN, coreKey);
		safeCallEE(parms->get(&coreType[c], "Type"));
		safeCallEE(parms->get(coreParentDesc[c], "Parents"));
		safeCallEE(parms->get(&coreParentConnType[c], "ParentsConnType", &coreParentsCnt[c]));
		//-- outputCnt is assumed to be the same across all Cores, and dependent on DataShape
		coreOutputCnt[c]=shape->predictionLen*shape->featuresCnt;

		safeCallEB(parms->restoreKey());
	}
	//-- 1.2. determine Layer for each Core, and cores count for each layer
	for (c=0; c<coresCnt; c++) {
		coreLayer[c]=getCoreLayer(c);
		layerCoresCnt[coreLayer[c]]++;
	}
	//-- 1.3. determine layersCnt, and InputCnt for each Core
	for (int l=0; l<MAX_ENGINE_LAYERS; l++) {
		for (c=0; c<layerCoresCnt[l]; c++) {
			if (l==0) {
				coreInputCnt[c]=shape->sampleLen*shape->featuresCnt;
			} else {
				coreInputCnt[c]=layerCoresCnt[l-1]*coreOutputCnt[c];
			}
		}
		if (c==0) break;
		layersCnt++;
	}
//----------------------------------------------

	safeCallEB(parms->setKey(parmKey));
	parms->get(&type, "Type");

	switch (type) {
	case ENGINE_CUSTOM:
		break;
	case ENGINE_WNN:
		safeCallEB(parms->setKey("WNN"));
		//... get() ...
		break;
	case ENGINE_XIE:
		safeCallEB(parms->setKey("XIE"));
		//... get() ...
		break;
	default:
		throwE("Invalid Engine Type: %d", 1, type);
		break;
	}

}
sEngine::~sEngine() {
	for (int i=0; i<coresCnt; i++) delete core[i];
	delete core;
}

void sEngine::addCore(tCoreLayout* coreLayout_) {
	switch (coreLayout_->type) {
	case CORE_NN:
		core[coresCnt]=new tNN(200,50,4,nullptr);
		coresCnt++;
		break;
	case CORE_GA: break;
	case CORE_SVM: break;
	case CORE_SOM: break;
	default:
		throwE("coreId %d : invalid coreType: %d", 2, coreLayout_->Id, coreLayout_->type);
		break;
	}
	coresCnt++;
}


//-- Engine Layout stuff
sEngineLayout::sEngineLayout(int coresCnt_) {
	coresCnt=coresCnt_;
	layerCoresCnt=(int*)malloc(MAX_ENGINE_LAYERS*sizeof(int)); for (int l=0; l<MAX_ENGINE_LAYERS; l++) layerCoresCnt[l]=0;

	coreLayer=(int*)malloc(coresCnt*sizeof(int));
	coreType=(int*)malloc(coresCnt*sizeof(int));
	coreDesc=(char**)malloc(coresCnt*sizeof(char*));
	coreParentsCnt=(int*)malloc(coresCnt*sizeof(int));
	coreParent=(int**)malloc(coresCnt*sizeof(int*));
	coreParentConnType=(int**)malloc(coresCnt*sizeof(int*));
	coreParentDesc=(char***)malloc(coresCnt*sizeof(char**));
	coreInputCnt=(int*)malloc(coresCnt*sizeof(int));
	coreOutputCnt=(int*)malloc(coresCnt*sizeof(int));
	for (int c=0; c<coresCnt; c++) {
		coreDesc[c]=(char*)malloc(XML_MAX_PARAM_VAL_LEN);
		coreParent[c]=(int*)malloc(coresCnt*sizeof(int));
		coreParentConnType[c]=(int*)malloc(coresCnt*sizeof(int));
		coreParentDesc[c]=(char**)malloc(coresCnt*sizeof(char*));
		for (int cc=0; cc<coresCnt; cc++) coreParentDesc[c][cc]=(char*)malloc(XML_MAX_PARAM_VAL_LEN);
	}
}
sEngineLayout::~sEngineLayout() {
	for (int c=0; c<coresCnt; c++) {
		free(coreDesc[c]);
		free(coreParent[c]);
		free(coreParentConnType[c]);
		for (int cc=0; cc<coresCnt; cc++) free(coreParentDesc[c][cc]);
		free(coreParentDesc[c]);
	}
	free(coreLayer);
	free(coreType);
	free(coreDesc);
	free(coreParent);
	free(coreParentsCnt);
	free(coreParentConnType);
	free(coreParentDesc);
	free(coreInputCnt);
	free(coreOutputCnt);
	free(layerCoresCnt);
}
int sEngineLayout::getCoreLayer(int c) {
	int ret=0;
	if (coreParentsCnt[c]==0) {
		ret=0;
	} else {
		for (int p=0; p<coreParentsCnt[c]; p++) {
			ret++;
			//coreParent[p]
		}
	}
	return ret;
}
