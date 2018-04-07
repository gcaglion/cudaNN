#include "Forecaster.h"

sForecaster::sForecaster(tDebugger* dbg_) {
	dbg=(dbg_==nullptr) ? (new tDebugger("Forecaster.err")) : dbg_;	//... handle specific debugger in xml ...
}
sForecaster::sForecaster(tParmsSource* parms, char* parmKey, tDebugger* dbg_) {
	dbg=(dbg_==nullptr) ? (new tDebugger("Forecaster.err")) : dbg_;

	//-- define forecaster Data
	safeCallEE(data=new tData(parms, ".Forecaster.Data", dbg))

	//-- define forecaster Engine, pass Data shape info
//	safeCallEE(engine=new tEngine(parms, ".Forecaster.Engine", data->shape));

	//-- define forecaster Persistor
	safeCallEE(persistor=new tLogger(parms, ".Forecaster.Persistor", dbg));

	//-- train each 
}
sForecaster::~sForecaster() {
	delete dbg;
	if (data!=nullptr) delete data;
	if (engine!=nullptr) delete engine;
	if (persistor!=nullptr) delete persistor;
}
