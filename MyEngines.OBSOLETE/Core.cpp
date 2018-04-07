#include "Core.h"

sCore::sCore(int type_, int inputCnt_, int outputCnt_, tDebugger* dbg_) {
	dbg=(dbg_==nullptr) ? (new tDebugger(new tFileInfo("Core.err"))) : dbg_;
	layout->inputCnt=inputCnt_; layout->outputCnt=outputCnt_;
}
sCore::sCore(tParmsSource* parms, tCoreLayout* layout_, tDebugger* dbg_) {
	dbg=(dbg_==nullptr) ? (new tDebugger(new tFileInfo("Core.err"))) : dbg_; //-- TO DO: How to handle specific <Debugger>/</Debugger> info??
	layout=layout_;
}
sCore::sCore() {}
sCore::~sCore() {}
