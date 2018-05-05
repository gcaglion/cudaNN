#include "BaseObj.h"

sBaseObj::sBaseObj(char* objName_, sBaseObj* objParent_, sDebuggerParms* dbgparms_) {
	try {
		strcpy_s(objName, 64, objName_);
		objParent=objParent_;
		stackLevel=(objParent==nullptr) ? 0 : objParent->stackLevel+1;
		dbg=new sDebugger(objName, dbgparms_);
		info("%s(%p)->%s() successful.", objName, this, __func__);
	}
	catch (std::exception exc) {
		char msg[DBG_MSG_MAXLEN]="";
		err_d("%s(%p)->%s() failed. Exception: %s", objName, this, __func__, exc.what());
		throw(exc);
	}
}
sBaseObj::~sBaseObj() {
	info("%s(%p)->%s() called.", objName, this, __func__);
	cleanup();
	info("\tDeleting dbg (%p) ...", dbg);
	delete dbg;
}

void sBaseObj::cleanup() {
	for (int c=0; c<childrenCnt; c++) {
		info("\tDeleting sub-object %s(%p) ...", child[c]->objName, child[c]);
		delete child[c];
	}
}

//-----------------------------------------------------------------------------------------------------------------------------------------------------
sDio::sDio(char* objName_, sBaseObj* objParent_, int prop1_, int prop2_, int children_, bool fail_, sDebuggerParms* dbgparms_) : sBaseObj(objName_, objParent_, dbgparms_) {
	prop1=prop1_; prop2=prop2_;

	if (children_>0) {
		safespawn(childDio1, sDio, 11, 11);
		safespawn(childDio2, sDio, 22, 22);
		safespawn(childDio3, sDio, 33, 33);
	}

	if (fail_) {
		fail("%s(%p)->%s(%d, %d) failed because of fail_=true", objName, this, __func__, prop1, prop2);
	}

	info("%s(%p)->%s(%d, %d) successful", objName, this, __func__, prop1, prop2);
}
void sDio::method(bool fail_) {

	if (fail_) {
		fail("%s(%p)->%s(%d, %d) failed because of fail_=true", objName, this, __func__, prop1, prop2);
	}
}
//-----------------------------------------------------------------------------------------------------------------------------------------------------

sRoot::sRoot(sDebuggerParms* rootdbgparms_) : sBaseObj("root", nullptr, rootdbgparms_) {

	//-----------------------------------------------
	//-- here we put everything that needs to be done
	//-----------------------------------------------

	try {

		tFileInfo* parmsFile;
		char* fname="c:/temp/parms.xml";
		spawn(parmsFile, tFileInfo, fname, FILE_MODE_READ);


		//-- 1. object creation (successful)
		safespawn(dio1, sDio, 1, 2);

		//-- 1.1. call to object method (success)
		safecall(dio1->method(false));

		//-- 2. object creation (constructor success)
		safespawn(dio2, sDio, -1, -2, 3);

		//-- 3.1. call to object method (success)
		safecall(dio2->method(false));
		//-- 3.2. call to object method (failure)
		//safecall(dio2->method(true));

		//-- 2. object creation (constructor success)
		safespawn(dio5, sDio, -1, -2);

		//-- 4. first object (constructor failure)
		safespawn(dio3, sDio, -10, -20, 0, true);

		//-- 5. first object (constructor successful)
		safespawn(dio4, sDio, 10, 20, 0, false);

	}
	catch (std::exception exc) {
		throw(exc);
	}
}