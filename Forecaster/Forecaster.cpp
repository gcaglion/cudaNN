#include "Forecaster.h"

sForecaster::sForecaster(tParmsSource* parms, char* parmKey, tDebugger* dbg_) : sBaseObj("Forecaster", dbg_) {

	//-- define forecaster Data
	safeCall(data=new tData(parms, ".Forecaster.Data"))

	//-- define forecaster Engine, pass Data shape info
	safeCall(engine=new tEngine(parms, ".Forecaster.Engine", data->shape));

	//-- define forecaster Persistor
	safeCall(persistor=new tLogger(parms, ".Forecaster.Persistor"));

	//-- train each 
}
sForecaster::~sForecaster() {
	delete persistor;
	delete engine;
	delete data;
}
