#include "Core.h"

sCore::sCore(int type_, tDebugger* dbg_) {
	dbg=(dbg_==nullptr) ? (new tDebugger(new tFileInfo("Core.err"))) : dbg_;
	type=type_;

}
sCore::sCore(tParmsSource* parms, char* parmKey, tDebugger* dbg_) {
	dbg=(dbg_==nullptr) ? (new tDebugger(new tFileInfo("Core.err"))) : dbg_; //-- TO DO: How to handle specific <Debugger>/</Debugger> info??

	safeCallEB(parms->setKey(parmKey));
	parms->get(&type, "Type");

	switch (type) {
	case CORE_NN:
		break;
	case CORE_GA:
		break;
	case CORE_SVM:
		break;
	case CORE_SOM:
		break;
	default:
		throwE("Invalid Core Type: %d", 1, type);
		break;
	}
}
sCore::~sCore() {}
