#include "Core.h"

void sCoreLayout::sCoreLayout_common(sDebuggerParms* dbgparms_, int Id_) {
	desc=(char*)malloc(CORE_MAX_DESC_LEN);
	parentId=(int*)malloc(CORE_MAX_PARENTS*sizeof(int));
	parentDesc=(char**)malloc(CORE_MAX_PARENTS*sizeof(char*)); for (int p=0; p<CORE_MAX_PARENTS; p++) parentDesc[p]=(char*)malloc(CORE_MAX_DESC_LEN);
	parentConnType=(int*)malloc(CORE_MAX_PARENTS*sizeof(int));

	char fname[MAX_PATH]; sprintf_s(fname, MAX_PATH, "Core%d.err", Id_);
}

sCoreLayout::sCoreLayout(char* objName_, sBaseObj* objParent_, tParmsSource* parms, int Id_, tDataShape* shape_, sDebuggerParms* dbgparms_) : sBaseObj(objName_, objParent_, dbgparms_) {
	sCoreLayout_common(dbgparms_, Id_);
	Id=Id_; shape=shape_;
	sprintf_s(desc, CORE_MAX_DESC_LEN, "Core%d", Id);

	safecall(parms->setKey(desc));
	safecall(parms->get(&type, "Type"));
	safecall(parms->get(parentDesc, "Parents"));
	safecall(parms->get(&parentConnType, "ParentsConnType", &parentsCnt));
	//-- extract parentId from each parentDesc
	for(int p=0; p<parentsCnt; p++){
		parentId[p]=atoi(right(parentDesc[p], (int)strlen(parentDesc[p])-4));
	}

}
sCoreLayout::~sCoreLayout() {
	free(desc);
	free(parentId);
	free(parentConnType);
	for (int p=0; p<CORE_MAX_PARENTS; p++) free(parentDesc[p]);
	free(parentDesc);
}

sCore::sCore(char* objName_, sBaseObj* objParent_, tParmsSource* parms, tCoreLayout* layout_, sDebuggerParms* dbgparms_) : sBaseObj(objName_, objParent_, dbgparms_) {
	layout=layout_;
}
sCore::sCore(char* objName_, sBaseObj* objParent_, tDataShape* baseShape_, sDebuggerParms* dbgparms_) : sBaseObj(objName_, objParent_, dbgparms_) {
	//-- TO DO !!! layout=new tCoreLayout(objName_, objParent_, baseShape_, dbgparms_);
}