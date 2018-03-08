#include "DataSource.h"


//=== sDataSource
sDataSource::sDataSource(int type_, bool calcBW_, int BWfeatureH_, int BWfeatureL_, tDbg* dbg_) {
	dbg=(dbg_==nullptr) ? (new tDbg(DBG_LEVEL_ERR, DBG_DEST_FILE, new tFileInfo("DataSource.err"))) : dbg_;
	type=type_; calcBW=calcBW_; BWfeatureH=BWfeatureH_; BWfeatureL=BWfeatureL_;
}
sDataSource::sDataSource(tParamMgr* parms, tDbg* dbg_) {
	dbg=(dbg_==nullptr) ? (new tDbg(DBG_LEVEL_ERR, DBG_DEST_FILE, new tFileInfo("DataSource.err"))) : dbg_;

	parms->getx(&type, "DataSourceType", true);

}
