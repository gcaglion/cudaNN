#include "DataSource.h"


//=== sDataSource
sDataSource::sDataSource(int type_, bool calcBW_, int BWfeatureH_, int BWfeatureL_, tDebugger* dbg_) : sBaseObj("DataSource", dbg_) {
	type=type_; calcBW=calcBW_; BWfeatureH=BWfeatureH_; BWfeatureL=BWfeatureL_;
}
sDataSource::sDataSource(tParmsSource* parms, tDebugger* dbg_) : sBaseObj("DataSource", dbg_) {
	parms->get(&type, "DataSourceType");
}
sDataSource::sDataSource(tDebugger* dbg_) : sBaseObj("DataSource", dbg_) {}
sDataSource::~sDataSource() {}