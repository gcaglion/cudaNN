#include "FXData.h"

//=== sFXData
void sFXData::sFXData_common(sDebuggerParms* dbgparms_){	//--parent DataSource properties
	Symbol[FX_SYMBOL_MAXLEN-1]='\0';
	TimeFrame[FX_TIMEFRAME_MAXLEN-1]='\0';

	type=FXDB_SOURCE;
	calcBW=true;
	//-- the following are fixed (OHLCV), and determined by loadOHLCV query
	featuresCnt=5;
	BWfeatureH=FXHIGH; BWfeatureL=FXLOW;
}
sFXData::sFXData(char* objName_, sBaseObj* objParent_, tDBConnection* db_, char* symbol_, char* tf_, bool isFilled_, sDebuggerParms* dbgparms_) : sDataSource(objName_, objParent_, dbgparms_) {
	sFXData_common(dbgparms_);
	//--
	db=db_;
	strcpy_s(Symbol, FX_SYMBOL_MAXLEN, symbol_);
	strcpy_s(TimeFrame, FX_TIMEFRAME_MAXLEN, tf_);
	IsFilled=isFilled_;
}
sFXData::sFXData(char* objName_, sBaseObj* objParent_, tParmsSource* parms, char* parmKey, sDebuggerParms* dbgparms_) : sDataSource(objName_, objParent_, dbgparms_) {
	sFXData_common(dbgparms_);
	//--
	safecall(parms->setKey(parmKey));
	parms->get(Symbol, "Symbol");
	parms->get(TimeFrame, "TimeFrame");
	parms->get(&IsFilled, "IsFilled");
	safespawn(db, tDBConnection, parms, objName);
}
sFXData::~sFXData() {}
