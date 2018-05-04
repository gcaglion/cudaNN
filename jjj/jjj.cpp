#include "..\FileInfo\FileInfo.h"

#define DEFAULT_DBG_FPATH "C:/temp/logs"
#define DEFAULT_DBG_FNAME "Debugger"
#define DEFAULT_DBG_VERBOSITY true
#define DBG_MSG_MAXLEN 1024
#define DBG_STACK_MAXLEN 32768

#define start(mask, ...) { \
	msgbld("%s->%s(%s) starting.", this, __func__, __VA_ARGS__); \
}

//-- this is only needed so we can pass parentdbg's object address to child's constructor
#define spawn(objname, objtype, ...){ \
	objname = new objtype(#objname, dbg, __VA_ARGS__); \
	child[childrenCnt]=objname; \
	childrenCnt++; \
}
#define safespawn(objname, objtype, ...){ \
	objtype* objname; \
	try { \
		spawn(objname, objtype, __VA_ARGS__); \
	} catch (std::exception exc) { \
		fail("%s->%s(%s) failed. Exception: %s", objName, __func__, #__VA_ARGS__, exc.what()); \
	} \
}

//-- info() , err(), fail() for sBaseObj object types
#define info(mask, ...) { if(dbg->verbose) err(mask, __VA_ARGS__); }
#define err(mask, ...) { \
	sprintf_s(dbg->msg, DBG_MSG_MAXLEN, mask, __VA_ARGS__); strcat_s(dbg->msg, DBG_MSG_MAXLEN, "\n"); \
	strcat_s(dbg->stackmsg, DBG_STACK_MAXLEN, dbg->msg); \
	if(parentdbg!=nullptr) sprintf_s(parentdbg->stackmsg, DBG_STACK_MAXLEN, "%s\n\t%s", parentdbg->stackmsg, dbg->stackmsg); \
	printf("%s", dbg->msg); \
	fprintf(dbg->outFile->handle, "%s", dbg->msg); \
}
#define fail(mask, ...) { \
	err(mask, __VA_ARGS__); \
	throw(std::exception(dbg->stackmsg)); \
}
//-- info() , err(), fail() for non-sBaseObj object types
#define info_d(mask, ...) { if(verbose) err_d(mask, __VA_ARGS__); }
#define err_d(mask, ...) { \
	sprintf_s(msg, DBG_MSG_MAXLEN, mask, __VA_ARGS__); \
	printf("%s\n", msg); \
}
#define fail_d(mask, ...) { \
	err_d(mask, __VA_ARGS__); \
	throw(std::exception(stackmsg)); \
}

struct sDebugger {
	bool verbose;
	tFileInfo* outFile;

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
			fail("%s(%p)->%s(%d, %d) failed because of fail_=true", objName_, this, __func__, prop1, prop2);
		}

		info("%s(%p)->%s(%d, %d) successful", objName_, this, __func__, prop1, prop2);
	}
};

struct sRoot : sBaseObj {

	//-- here we put everything that needs to be done
	sRoot(sDebugger* parentdbg_, bool verbose_) : sBaseObj("root", parentdbg_, verbose_) {

		//-- 1. first object (successful)
		safespawn(dio1, sDio, 1, 2, false);
		//-- 2. first object (constructor success)
		safespawn(dio2, sDio, -1, -2, false);
		//-- 3. first object (constructor failure)
/*
		sDio* dio3; 
		try {
			spawn(dio3, sDio, -10,-20,true); 
		}
		catch (std::exception exc) {
				printf("%s\n", exc.what());
			fail("%s->%s(%s) failed. Exception: %s", "dio3", __func__, "-10,-20,true", exc.what()); 
		} 
*/
		safespawn(dio3, sDio, -10, -20, true);
		//-- 4. first object (constructor successful)
		safespawn(dio4, sDio, 10, 20, false);

	}
	~sRoot() {}

};

#define clientFail(){ \
	delete root; \
	printf("Client main() failed. \n"); \
	system("pause"); \
	return -1; \
}
#define clientSuccess(){ \
	delete root; \
	printf("Client main() success. \n"); \
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
		printf("Critical Error: Could not create main debugger.\n");
		clientFail();
	}

	//-- 1. create root object, pass root debugger. root constructor does everything else
	try {
		root=new sRoot(dbg, verbose_);
	} catch(std::exception exc){
		clientFail();
	}

	clientSuccess();
}