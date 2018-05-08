#include "FileData.h"

//=== sFileData
sFileData::sFileData(char* objName_, s0* objParent_, tFileInfo* srcFile_, int fieldSep_, bool calcBW_, int BWfeatureH_, int BWfeatureL_, sDebuggerParms* dbgparms_) : sDataSource(objName_, objParent_, dbgparms_) {
	//--parent DataSource properties
	type=FILE_SOURCE;
	calcBW=calcBW_; BWfeatureH=BWfeatureH_; BWfeatureL=BWfeatureL_;
	//--
	srcFile=srcFile_; fieldSep=fieldSep_;
	//-- need to set featuresCnt, but we need to red file to do that!!!
	featuresCnt=-99;
}
sFileData::sFileData(char* objName_, s0* objParent_, tParmsSource* parms, char* parmKey, sDebuggerParms* dbgparms_) : sDataSource(objName_, objParent_, dbgparms_) {
	safecall(parms->setKey(parmKey));
	char ffname[MAX_PATH];
	safecall(srcFile=new tFileInfo(ffname, FILE_MODE_READ));
	safecall(parms->get(&fieldSep, "FieldSep"));
}
sFileData::~sFileData(){}
