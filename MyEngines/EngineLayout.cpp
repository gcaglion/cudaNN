#include "EngineLayout.h"

sEngineLayout::sEngineLayout(int coresCnt_) {
	coresCnt=coresCnt_;
	layerCoresCnt=(int*)malloc(MAX_ENGINE_LAYERS*sizeof(int)); for (int l=0; l<MAX_ENGINE_LAYERS; l++) layerCoresCnt[l]=0;
	//core=(tCore**)malloc(coresCnt*sizeof(tCore*));
	coreLayer=(int*)malloc(coresCnt*sizeof(int));
	coreType=(int*)malloc(coresCnt*sizeof(int));
	coreParentsCnt=(int*)malloc(coresCnt*sizeof(int));
	coreParent=(int**)malloc(coresCnt*sizeof(int*));
	coreParentConnType=(int**)malloc(coresCnt*sizeof(int*));
	coreParentDesc=(char***)malloc(coresCnt*sizeof(char**));
	coreInputCnt=(int*)malloc(coresCnt*sizeof(int));
	coreOutputCnt=(int*)malloc(coresCnt*sizeof(int));
	for (int c=0; c<coresCnt; c++) {
		coreParent[c]=(int*)malloc(coresCnt*sizeof(int));
		coreParentConnType[c]=(int*)malloc(coresCnt*sizeof(int));
		coreParentDesc[c]=(char**)malloc(coresCnt*sizeof(char*));
		for (int cc=0; cc<coresCnt; cc++) coreParentDesc[c][cc]=(char*)malloc(XML_MAX_PARAM_VAL_LEN);
	}
}
sEngineLayout::~sEngineLayout() {
	for (int c=0; c<coresCnt; c++) {
		free(coreParent[c]);
		free(coreParentConnType[c]);
		for (int cc=0; cc<coresCnt; cc++) free(coreParentDesc[c][cc]);
		free(coreParentDesc[c]);
	}
	free(coreLayer);
	free(coreType);
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

