#include "Core.h"

sCore::sCore(int type_, int inputCnt_, int outputCnt_, tDebugger* dbg_) {
	dbg=(dbg_==nullptr) ? (new tDebugger(new tFileInfo("Core.err"))) : dbg_;

}
sCore::sCore(tParmsSource* parms, tCoreLayout* layout_, tDebugger* dbg_) {
	dbg=(dbg_==nullptr) ? (new tDebugger(new tFileInfo("Core.err"))) : dbg_; //-- TO DO: How to handle specific <Debugger>/</Debugger> info??
	layout=layout_;

	int kaz;
	switch (layout->type) {
	case CORE_NN:
		kaz=0;
		break;
	case CORE_GA:
		break;
	case CORE_SVM:
		break;
	case CORE_SOM:
		break;
	default:
		throwE("Invalid Core Type: %d", 1, layout->type);
		break;
	}
}
sCore::sCore() {}
sCore::~sCore() {}
