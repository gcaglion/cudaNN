#include "FXData.h"

//=== sFXData
void sFXData::sFXData_common(tDebugger* dbg_){	//--parent DataSource properties
	dbg=(dbg_==nullptr) ? (new tDebugger(new tFileInfo("FXData.err"))) : dbg_;
	type=FXDB_SOURCE;
	calcBW=true;
	//-- the following are fixed (OHLCV), and determined by loadOHLCV query
	featuresCnt=5;
	BWfeatureH=FXDATA_HIGH; BWfeatureL=FXDATA_LOW;
}
sFXData::sFXData(tDBConnection* db_, char* symbol_, char* tf_, int isFilled_, tDebugger* dbg_) {
	sFXData_common(dbg_);
	//--
	db=db_;
	strcpy_s(Symbol, FX_SYMBOL_MAX_LEN, symbol_);
	strcpy_s(TimeFrame, FX_TIMEFRAME_MAX_LEN, tf_);
	IsFilled=isFilled_;
}
sFXData::sFXData(tParmsSource* parms, tDebugger* dbg_) {
	sFXData_common(dbg_);
	//--
	parms->get(Symbol, "Symbol");
	parms->get(TimeFrame, "TimeFrame");
	parms->get(&IsFilled, "IsFilled");
	safeCallEB(parms->setKey("DBConnection"));
	safeCallEE(db=new tDBConnection(parms));
}

