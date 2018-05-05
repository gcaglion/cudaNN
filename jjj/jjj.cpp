#include "jjj.h"

struct sDebuggerParms {
	bool verbose;
	bool timing;
	bool pauseOnError;

	sDebuggerParms(bool verbose_=DEFAULT_DBG_VERBOSITY, bool timing_=DEFAULT_DBG_TIMING, bool pauseOnError_=DEFAULT_DBG_PAUSERR){
		verbose=verbose_;
		timing=timing_;
		pauseOnError=pauseOnError_;
	}
};

struct sDebugger {

	sDebuggerParms* parms;

	tFileInfo* outFile;
	char msg[DBG_MSG_MAXLEN]="";
	char stackmsg[DBG_STACK_MAXLEN]="";

	sDebugger(char* outFileName=DEFAULT_DBG_FNAME, sDebuggerParms* parms_=nullptr, char* outFilePath=DEFAULT_DBG_FPATH) {
		if (parms_==nullptr) {
			parms=new sDebuggerParms();
		} else {
			parms=parms_;
		}
		char outfname[MAX_PATH];
		sprintf_s(outfname, MAX_PATH, "%s/%s(%p).%s", outFilePath, outFileName, this, (parms->verbose) ? "log": "err");
		try {
			outFile=new tFileInfo(outfname, FILE_MODE_WRITE);
			//info_d("sDebugger(%p)->%s() called. Successfully created debugger outFile %s ...\n", this, __func__, outfname);
		}
		catch (std::exception exc) {			
			err_d("sDebugger(%p)->%s() failed. Error creating debugger outFile %s ...\n", this, __func__, outfname);
			throw(exc);
		}
	}
	~sDebugger() {
		//info_d("sDebugger(%p)->%s() called. Deleting %s ...", this, __func__, outFile->FullName);
		delete outFile;
	}

	void stackup(const char* msg) {
		strcat_s(stackmsg, DBG_STACK_MAXLEN, msg);
	}

};

#define BASEOBJ_MAX_CHILDREN 64
struct sBaseObj {

	char objName[64];
	sBaseObj* objParent;
	int stackLevel;
	int childrenCnt=0;
	sBaseObj* child[BASEOBJ_MAX_CHILDREN];

	sDebugger* dbg;

	sBaseObj(char* objName_, sBaseObj* objParent_, sDebuggerParms* dbgparms_) {
		try {
			strcpy_s(objName, 64, objName_);
			objParent=objParent_;
			stackLevel=(objParent==nullptr) ? 0 : objParent->stackLevel+1;
			dbg=new sDebugger(objName, dbgparms_);
			info("%s(%p)->%s() successful.", objName, this, __func__);
		} catch(std::exception exc) {
			char msg[DBG_MSG_MAXLEN]="";
			err_d("%s(%p)->%s() failed. Exception: %s", objName, this, __func__, exc.what());
			throw(exc);
		}
	}

	~sBaseObj() {
		info("%s(%p)->%s() called.", objName, this, __func__);
		cleanup();
		info("\tDeleting dbg (%p) ...", dbg);
		delete dbg;
	}

	void cleanup() {
		for (int c=0; c<childrenCnt; c++) {
			info("\tDeleting sub-object %s(%p) ...", child[c]->objName, child[c]);
			delete child[c];
		}
	}

};

struct sDio : sBaseObj {
	int prop1;
	int prop2;

	sDio* childDio1;
	sDio* childDio2;
	sDio* childDio3;

	sDio(char* objName_, sBaseObj* objParent_, int prop1_, int prop2_, int children_=0, bool fail_=false, sDebuggerParms* dbgparms_=nullptr) : sBaseObj(objName_, objParent_, dbgparms_) {
		prop1=prop1_; prop2=prop2_;

		if(children_>0){
			safespawn(childDio1, sDio, 11, 11);
			safespawn(childDio2, sDio, 22, 22);
			safespawn(childDio3, sDio, 33, 33);
		}

		if (fail_) {
			fail("%s(%p)->%s(%d, %d) failed because of fail_=true", objName, this, __func__, prop1, prop2);
		}

		info("%s(%p)->%s(%d, %d) successful", objName, this, __func__, prop1, prop2);
	}

	void method(bool fail_) {

		if (fail_) {
			fail("%s(%p)->%s(%d, %d) failed because of fail_=true", objName, this, __func__, prop1, prop2);
		}
	}
};

struct sRoot : sBaseObj {

	//-- here we put everything that needs to be done
	sRoot(sDebuggerParms* rootdbgparms_=nullptr) : sBaseObj("root", nullptr, rootdbgparms_) {

		try {

			tFileInfo* parmsFile;
			char* fname="C:/temp/parms.xml";

			safecall(spawnFile(parmsFile, fname, FILE_MODE_READ));

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

};

#define clientFail(failmsg){ \
	delete root; \
	printf("Client failed: %s\n", failmsg); \
	system("pause"); \
	return -1; \
}
#define clientSuccess(){ \
	delete root; \
	printf("Client success. \n"); \
	system("pause"); \
	return 0; \
}

int main(int argc, char* argv[]) {
	
	//-- 1. create root object. root constructor does everything else
	sRoot* root=nullptr;
	try {
		root=new sRoot();	//-- always takes default debugger settings
	} catch(std::exception exc){
		clientFail("Exception thrown by root. See stack.");
	}

	clientSuccess();
}