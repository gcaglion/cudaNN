#include "MT4Data.h"

//=== sMT4Data
void sMT4Data::sMT4Data_common(tDebugger* dbg_){	//--parent DataSource properties
	dbg=(dbg_==nullptr) ? (new tDebugger("MT4Data.err")) : dbg_;
	type=MT4_SOURCE;
	calcBW=true;
	//-- the following are fixed (OHLCV), and determined by loadOHLCV query
	featuresCnt=5;
	BWfeatureH=1; BWfeatureL=2;
}
sMT4Data::sMT4Data(tParmsSource* parms, tDebugger* dbg_) {
	sMT4Data_common(dbg_);
}

//=== sMT4Data
sMT4Data::sMT4Data(int accountId_, tDebugger* dbg_) {
	sMT4Data_common(dbg_);
	//--
	accountId=accountId_;
}

