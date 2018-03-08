#include "MT4Data.h"

//=== sMT4Data
void sMT4Data::sMT4Data_common(tDbg* dbg_){	//--parent DataSource properties
	dbg=(dbg_==nullptr) ? (new tDbg(DBG_LEVEL_ERR, DBG_DEST_FILE, new tFileInfo("MT4Data.err"))) : dbg_;
	type=SOURCE_DATA_FROM_MT4;
	calcBW=true;
	//-- the following are fixed (OHLCV), and determined by loadOHLCV query
	featuresCnt=5;
	BWfeatureH=1; BWfeatureL=2;
}
sMT4Data::sMT4Data(tParamMgr* parms, tDbg* dbg_) {
	sMT4Data_common(dbg_);
}

//=== sMT4Data
sMT4Data::sMT4Data(int accountId_, tDbg* dbg_) {
	sMT4Data_common(dbg_);
	//--
	accountId=accountId_;
}

