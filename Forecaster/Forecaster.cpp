#include "Forecaster.h"

sForecaster::sForecaster(char* objName_, s0* objParent_, tParmsSource* parms, char* parmKey, sDebuggerParms* dbgparms_) : s0(objName_, objParent_, dbgparms_) {

	tData* data;
	tEngine* engine;
	tLogger* persistor;

	//-- define forecaster Data
	safespawn(data, tData, parms, ".Forecaster.Data");

	//-- define forecaster Engine, pass Data shape info

	safespawn(engine, tEngine, parms, ".Forecaster.Engine", data->shape);

	//-- define forecaster Persistor
	safespawn(persistor, tLogger, parms, ".Forecaster.Persistor");

	//-- train each 
}
