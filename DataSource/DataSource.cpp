#include "DataSource.h"


//=== sDataSource
sDataSource::sDataSource(char* objName_, sBaseObj* objParent_, int type_, bool calcBW_, int BWfeatureH_, int BWfeatureL_, sDebuggerParms* dbgparms_) : sBaseObj(objName_, objParent_, dbgparms_) {
	type=type_; calcBW=calcBW_; BWfeatureH=BWfeatureH_; BWfeatureL=BWfeatureL_;
}
sDataSource::sDataSource(char* objName_, sBaseObj* objParent_, tParmsSource* parms, sDebuggerParms* dbgparms_) : sBaseObj(objName_, objParent_, dbgparms_) {
	parms->get(&type, "Type");
}
sDataSource::sDataSource(char* objName_, sBaseObj* objParent_, sDebuggerParms* dbgparms_) : sBaseObj(objName_, objParent_, dbgparms_) {}
sDataSource::~sDataSource() {}