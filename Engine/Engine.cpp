#include "Engine.h"

//-- Engine stuff
void sEngine::sEngine_common(tParmsSource* parms, tDataShape* shape_, tDebugger* dbg_) {
	dbg=(dbg_==nullptr) ? (new tDebugger("Engine.err")) : dbg_;	//... handle specific debugger in xml ...
	layerCoresCnt=(int*)malloc(MAX_ENGINE_LAYERS*sizeof(int)); for (int l=0; l<MAX_ENGINE_LAYERS; l++) layerCoresCnt[l]=0;
	shape=shape_;
}
sEngine::sEngine(tParmsSource* parms, char* parmKey, tDataShape* shape_, tDebugger* dbg_) {
	int c;

	sEngine_common(parms, shape_, dbg_);

	safeCallEB(parms->setKey(parmKey));
	safeCallEB(parms->backupKey());

	parms->get(&type, "Type");

	switch (type) {
	case ENGINE_CUSTOM:
		safeCallEB(parms->setKey("Custom"));
		//-- 0. coresCnt
		safeCallEE(parms->get(&coresCnt, "CoresCount"));
		//-- 1. malloc one core and one coreLayout for each core
		core=(tCore**)malloc(coresCnt*sizeof(tCore*));
		coreLayout=(tCoreLayout**)malloc(coresCnt*sizeof(tCoreLayout*));
		//-- 2. create layout, set base coreLayout properties for each Core (type, desc, connType, outputCnt)
		for (c=0; c<coresCnt; c++) {
			safeCallEB(parms->backupKey());
			coreLayout[c]=new tCoreLayout(parms, c, shape);
			safeCallEB(parms->restoreKey());
		}
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

	//-- 3. once all coreLayouts are created (and all  parents are set), we can determine Layer for each Core, and cores count for each layer
	for (c=0; c<coresCnt; c++) {
		setCoreLayer(coreLayout[c]);
		layerCoresCnt[coreLayout[c]->layer]++;
	}
	//-- 4. determine layersCnt, and InputCnt for each Core
	for (int l=0; l<MAX_ENGINE_LAYERS; l++) {
		for (c=0; c<layerCoresCnt[l]; c++) {
			if (l==0) {
				//-- do nothing. keep core shape same as engine shape
			} else {
				//-- change sampleLen
				coreLayout[c]->shape->sampleLen=layerCoresCnt[l-1]*coreLayout[c]->shape->predictionLen;
			}
		}
		if (c==0) break;
		layersCnt++;
	}

	//-- 5. add each core
	for (c=0; c<coresCnt; c++) {
		safeCallEE(parms->backupKey());
		safeCallEE(addCore(parms, c));
		safeCallEE(parms->restoreKey());
	}
}
sEngine::~sEngine() {
	for (int i=0; i<coresCnt; i++) delete core[i];
	delete core;
	free(layerCoresCnt);
	delete dbg;
}


void sEngine::addCore(tParmsSource* parms, int coreId) {

	safeCallEE(parms->setKey(coreLayout[coreId]->desc));

	switch (coreLayout[coreId]->type) {
	case CORE_NN:
		safeCallEE(core[coresCnt]=new tNN(parms, coreLayout[coreId]));
		break;
	case CORE_GA: break;
	case CORE_SVM: break;
	case CORE_SOM: break;
	default:
		throwE("coreId %d : invalid coreType: %d", 2, coreLayout[coreId]->Id, coreLayout[coreId]->type);
		break;
	}
}

void sEngine::setCoreLayer(tCoreLayout* c){
	int ret=0;
	int maxParentLayer=-1;
	for (int p=0; p<c->parentsCnt; p++) {
		tCoreLayout* parent=coreLayout[c->parentId[p]];
		setCoreLayer(parent);
		if (parent->layer>maxParentLayer) {
			maxParentLayer=parent->layer;
		}
		ret=maxParentLayer+1;
	}
	c->layer=ret;
}

//-- Engine Layout stuff
/*sEngineLayout::sEngineLayout(int coresCnt_) {
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
*/