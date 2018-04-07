#include "DataSource.h"


//=== sDataSource
sDataSource::sDataSource(){}
sDataSource::sDataSource(int type_, bool calcBW_, int BWfeatureH_, int BWfeatureL_, tDebugger* dbg_) {
	dbg=(dbg_==nullptr) ? (new tDebugger("DataSource.err")) : dbg_;
	type=type_; calcBW=calcBW_; BWfeatureH=BWfeatureH_; BWfeatureL=BWfeatureL_;
}
sDataSource::sDataSource(tParmsSource* parms, tDebugger* dbg_) {
	dbg=(dbg_==nullptr) ? (new tDebugger("DataSource.err")) : dbg_;

	parms->get(&type, "DataSourceType");

}
sDataSource::~sDataSource() {
	delete dbg;
}