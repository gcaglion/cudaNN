#include "DataSource.h"


//=== sDataSource
sDataSource::sDataSource(char* objName_, s0* objParent_, int type_, bool calcBW_, int BWfeatureH_, int BWfeatureL_, sDebuggerParms* dbgparms_) : s0(objName_, objParent_, dbgparms_) {
	type=type_; calcBW=calcBW_; BWfeatureH=BWfeatureH_; BWfeatureL=BWfeatureL_;
}
sDataSource::sDataSource(char* objName_, s0* objParent_, tParmsSource* parms, sDebuggerParms* dbgparms_) : s0(objName_, objParent_, dbgparms_) {
	parms->get(&type, "Type");
}
sDataSource::sDataSource(char* objName_, s0* objParent_, sDebuggerParms* dbgparms_) : s0(objName_, objParent_, dbgparms_) {}
sDataSource::~sDataSource() {}