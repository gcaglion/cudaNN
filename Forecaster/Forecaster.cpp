#include "Forecaster.h"

sForecaster::sForecaster(tDebugger* dbg_) {
	dbg=(dbg_==nullptr) ? (new tDebugger("Forecaster.err")) : dbg_;	//... handle specific debugger in xml ...
}
sForecaster::sForecaster(tParmsSource* parms, char* parmKey, tDebugger* dbg_) {
	dbg=(dbg_==nullptr) ? (new tDebugger("Forecaster.err")) : dbg_;

	//-- define forecaster Data
	safeCallEE(data=new tData(parms, ".Forecaster.Data"))

	//-- define forecaster Engine, pass Data shape info
	safeCallEE(engine=new tEngine(parms, ".Forecaster.Engine", data->shape));

	//-- define forecaster Persistor
	safeCallEE(persistor=new tLogger(parms, ".Forecaster.Persistor"));

	//-- train each 
}
sForecaster::~sForecaster() {
	delete persistor;
	delete engine;
	delete data;
	delete dbg;
}
