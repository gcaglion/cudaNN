#include "FXData.h"

//=== sFXData
void sFXData::sFXData_common(tDebugger* dbg_){	//--parent DataSource properties
	dbg=(dbg_==nullptr) ? (new tDebugger(new tFileInfo("FXData.err"))) : dbg_;
	type=FXDB_SOURCE;
	calcBW=true;
	//-- the following are fixed (OHLCV), and determined by loadOHLCV query
	featuresCnt=5;
	BWfeatureH=FXHIGH; BWfeatureL=FXLOW;
}
sFXData::sFXData(tDBConnection* db_, char* symbol_, char* tf_, bool isFilled_, tDebugger* dbg_) {
	sFXData_common(dbg_);
	//--
	db=db_;
	strcpy_s(Symbol, FX_SYMBOL_MAX_LEN, symbol_);
	strcpy_s(TimeFrame, FX_TIMEFRAME_MAX_LEN, tf_);
	IsFilled=isFilled_;
}
sFXData::sFXData(tParmsSource* parms, char* parmKey, tDebugger* dbg_) {
	sFXData_common(dbg_);
	//--
	safeCallEB(parms->setKey(parmKey));
	parms->get(Symbol, "Symbol");
	parms->get(TimeFrame, "TimeFrame");
	parms->get(&IsFilled, "IsFilled");
	safeCallEE(db=new tDBConnection(parms, "DBConnection"));
}

