#include "jjj.h"

struct sDebugger {
	bool verbose;
	tFileInfo* outFile;

	int stackLevel;
	char msg[DBG_MSG_MAXLEN]="";
	char stackmsg[DBG_STACK_MAXLEN]="";

	sDebugger(char* outFileName=DEFAULT_DBG_FNAME, bool verbose_=DEFAULT_DBG_VERBOSITY, char* outFilePath=DEFAULT_DBG_FPATH) {
		verbose=verbose_;
		char outfname[MAX_PATH];
		sprintf_s(outfname, MAX_PATH, "%s/%s(%p).%s", outFilePath, outFileName, this, (verbose) ? "log": "err");
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

};

#define BASEOBJ_MAX_CHILDREN 64
struct sBaseObj {

	char objName[64];
	int childrenCnt=0;
	sBaseObj* objecttAddr=this;
	sBaseObj* child[BASEOBJ_MAX_CHILDREN];

	sDebugger* dbg;
	sDebugger* parentdbg;

	sBaseObj(char* objName_, sDebugger* parentdbg_, bool verbose_=DEFAULT_DBG_VERBOSITY) {
		try {
			strcpy_s(objName, 64, objName_);
			parentdbg=parentdbg_;
			dbg=new sDebugger(objName);
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

	sDio(char* objName_, sDebugger* parentdbg_, int prop1_, int prop2_, bool fail_=false, bool verbose_=DEFAULT_DBG_VERBOSITY) : sBaseObj(objName_, parentdbg_, verbose_) {
		prop1=prop1_; prop2=prop2_;

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
	sRoot(sDebugger* parentdbg_, bool verbose_) : sBaseObj("root", parentdbg_, verbose_) {

		//-- 1. object creation (successful)
		safespawn(dio1, sDio, 1, 2, false);

		//-- 1.1. call to object method (success)
		//safecall(dio1->method(false));

		//-- 2. object creation (constructor success)
		safespawn(dio2, sDio, -1, -2, false);

		//-- 3.1. call to object method (success)
		safecall(dio2->method(false));
		//-- 3.2. call to object method (failure)
		try {
			dio2->method(true);
		}
		catch (std::exception exc) {
			//-- fail()
			sprintf_s(dbg->msg, DBG_MSG_MAXLEN, "dioporco failed"); strcat_s(dbg->msg, DBG_MSG_MAXLEN, "\n"); 
			strcat_s(dbg->stackmsg, DBG_STACK_MAXLEN, dbg->msg); 
			if (parentdbg!=nullptr) sprintf_s(parentdbg->stackmsg, DBG_STACK_MAXLEN, "%s\nt%s", parentdbg->stackmsg, dbg->msg); 
			printf("%s", dbg->msg); 
			fprintf(dbg->outFile->handle, "%s", dbg->msg); 
		}
		safecall(dio2->method(true));

		//-- 2. object creation (constructor success)
		safespawn(dio5, sDio, -1, -2, false);

		//-- 4. first object (constructor failure)
		safespawn(dio3, sDio, -10, -20, false);

		//-- 5. first object (constructor successful)
		safespawn(dio4, sDio, 10, 20, false);

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
	
	//-- root object and debugger
	sRoot* root=nullptr;
	sDebugger* dbg=nullptr;
	bool verbose_=true;

	//-- 1. create root debugger
	try {
		dbg=new sDebugger("mainDebugger", verbose_);
	} catch (std::exception exc) {
		clientFail("Critical Error: Could not create main debugger.");
	}

	//-- 1. create root object, pass root debugger. root constructor does everything else
	try {
		root=new sRoot(dbg, verbose_);
	} catch(std::exception exc){
		clientFail("Exception thrown by root. See stack.");
	}

	clientSuccess();
}