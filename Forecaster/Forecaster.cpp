#include "Forecaster.h"

sForecaster::sForecaster(tDebugger* dbg_) {
	dbg=(dbg_==nullptr) ? (new tDebugger(new tFileInfo("Forecaster.err"))) : dbg_;	//... handle specific debugger in xml ...
}
sForecaster::sForecaster(tParmsSource* parms, char* parmKey, tDebugger* dbg_) {
	dbg=(dbg_==nullptr) ? (new tDebugger(new tFileInfo("Forecaster.err"))) : dbg_;

	//-- define forecaster Data
	safeCallEE(data=new tData(parms, ".Forecaster.Data", dbg))

	//-- define forecaster Engine
	safeCallEE(parms->setKey(".Forecaster.Engine"));

	//-- define forecaster Persistor
	safeCallEE(parms->setKey(".Forecaster.Persistor"));

}
sForecaster::~sForecaster() {
	delete dbg;
	if (data!=nullptr) delete data;
	if (engine!=nullptr) delete engine;
	if (persistor!=nullptr) delete persistor;
}
