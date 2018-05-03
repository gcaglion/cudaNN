#include "../CommonEnv00.h"
#include "../FileInfo/FileInfo.h"

//#include "kazM2.h"

#define DBG_DEFAULT_VERBOSE true
#define DBG_ERRMSG_SIZE 32768
#define MAX_OBJ_CHILDREN 32

#define mainSuccess()	{ printf("main() success.\n"); system("pause"); return -1; } 
#define mainFail(...)	{ printf(__VA_ARGS__); system("pause"); return  0; }
//
#define msgbld(mask, ...) { sprintf_s(dbg->msg, DBG_ERRMSG_SIZE, mask, __VA_ARGS__); }
#define info(mask, ...) {}
#define start(mask, ...) {}
#define failC(mask, ...) { msgbld(mask, __VA_ARGS__); throw std::exception(((sLbase*)parentObj)->dbg->stackmsg); }
#define success() { msgbld("%s->%s() Success.", objName, __func__); dbg->out(); }
#define nakedNew(cname, ctype, ...) { cname = new ctype(this, #cname, __VA_ARGS__); }

struct sDebuger {
	int stackLevel;
	bool verbose;
	tFileInfo* outfile;

	char stackmsg[DBG_ERRMSG_SIZE]="";
	char msg[DBG_ERRMSG_SIZE]="";

	sDebuger(char* outfilename, bool verbose_=false) {
		verbose=verbose_;
		outfile=new tFileInfo(outfilename, "C:/temp/logs", FILE_MODE_WRITE);
	}
	~sDebuger() {
		printf("%s() called on %p. Deleting outFile (%s) ...\n", __func__, this, outfile->FullName);
		delete outfile;
	}

	void out(bool wholeStack=false) {
		if (wholeStack) {
			printf("%s\n", stackmsg);
			fprintf(outfile->handle, "%s\n", stackmsg);
		} else {
			printf("%s\n", msg);
			fprintf(outfile->handle, "%s\n", msg);
			sprintf_s(stackmsg, DBG_ERRMSG_SIZE, "%s\n\t%s", stackmsg, msg);
		}
	}

};

struct sLbase {
	char objName[MAX_PATH]="";
	char dbgfname[MAX_PATH]="";

	int	 stackLevel=0;
	void*	parentObj=nullptr;
	int		subObjCnt=0;
	void**	subObj;

	sDebuger* dbg;

	void setParent(void* parentObj_) {
		parentObj=parentObj_;
		if (parentObj!=nullptr) stackLevel=((sLbase*)parentObj)->stackLevel+1;
	}
	void setDebugger(sDebuger* dbg_, bool verbose_){
		//-- if we didn't pass a valid dbg_, regardless of what this new object does, it will always create its own debugger
		if (dbg_==nullptr) {
			sprintf_s(dbgfname, MAX_PATH, "%s.%s", objName, ((verbose_) ? "log" : "err"));
			try {
				dbg=new sDebuger(dbgfname, verbose_);
				dbg->stackLevel=stackLevel;
				msgbld("new sDebugger(%s) completed successfully. dbg->stackLevel=%d", objName, dbg->stackLevel)
				dbg->out();
			}
			catch (std::exception exc) {
				char errtmp[DBG_ERRMSG_SIZE]="";
				sprintf_s(errtmp, DBG_ERRMSG_SIZE, "%s() failed to create its own debugger. Exception=%s \n", objName, exc.what());
				cleanup(errtmp);
			}
		} else {
			dbg=dbg_;
		}
	}

	sLbase(void* parentObj_, char* objName_, bool verbose_=DBG_DEFAULT_VERBOSE, sDebuger* dbg_=nullptr) {
		subObj=(void**)malloc(MAX_OBJ_CHILDREN*sizeof(void*));
		sprintf_s(objName, MAX_PATH, "%s(%p)", objName_, this);
		setDebugger(dbg_, verbose_);
		parentObj=parentObj_;
		if (parentObj!=nullptr) stackLevel=((sLbase*)parentObj)->stackLevel+1;
	}

	//-- this constructor is only called by root object in client
	sLbase(sDebuger* rootdbg_, bool verbose_=DBG_DEFAULT_VERBOSE) {
		subObj=(void**)malloc(MAX_OBJ_CHILDREN*sizeof(void*));
		sprintf_s(objName, MAX_PATH, "%s(%p)", "root", this);
		//-- rootdbg must be valid.
		stackLevel=0;
		setDebugger(rootdbg_, verbose_);
	}


	void cleanup(char* reason) {

		info("%s->cleanup() called; reason=%s", objName, reason);
		for (int o=0; o<subObjCnt; o++) {
			info("%s->cleanup() calling %s->cleanup() ...", objName, ((sLbase*)subObj[o])->objName);
			((sLbase*)subObj[o])->cleanup("called from parent cleanup()");
		}
		info("%s->cleanup() calling (delete dbg) ...", objName);
		//-- append dbg->stackmsg to parent's before suicide
		if (parentObj!=nullptr) {
			sprintf_s(((sLbase*)parentObj)->dbg->stackmsg, DBG_ERRMSG_SIZE, "%s\n\t%s", ((sLbase*)parentObj)->dbg->stackmsg, dbg->stackmsg);
			delete dbg;
		}
	}
	~sLbase() {
		printf("%s->%s() called. calling cleanup()..", objName, __func__);
		cleanup("destructor call");
	}
};
/*
struct sTimeSerie :sLbase {
	int tsprop1;
	int tsprop2;

	sTimeSerie(char* objName_, int tsprop1_, int tsprop2_, sDebuger* dbg_=nullptr) :sLbase(this, objName_, dbg_) {
		tsprop1=tsprop1_;
		tsprop2=tsprop2_;



		start("tsprop1=%d ; tsprop2=%d", tsprop1, tsprop2);
		//-- ... do specific constructor stuff ...

		//-- constructor failure
		if (tsprop1!=1) failC("tsprop1=%d ; tsprop2=%d", tsprop1, tsprop2);

		success();
	}

	void dump() {
	}

};

struct sForecaster :sLbase {
	int prop1;
	int prop2;
	sTimeSerie* ts1=nullptr;
	sTimeSerie* ts2=nullptr;

	sForecaster(char* objName_, int prop1_, int prop2_, sDebuger* dbg_=nullptr) :sLbase(this, objName_, dbg_) {
		prop1=prop1_;
		prop2=prop2_;

		start("prop1=%d ; prop2=%d", prop1, prop2);
		//-- ... do specific constructor stuff ...

		//-- create child ts1 (success)
		newC(ts1, sTimeSerie("TrainTimeSerie", 1, -1));

		//-- create child ts2 (fail)
		//newC(ts2, sTimeSerie("TestTimeSerie", 2, -2));
		try {
				info("Trying: %s = %s ...", "ts2", "new ts2(... blah ...)");
				ts2 = new sTimeSerie("TestTimeSerie", 2, -2);
				subObj[subObjCnt]=ts2;
				subObjCnt++;
				info("%s = %s completed successfully.", "ts2", "new ts2(... blah ...)");
		}
		catch (std::exception exc) {
			err("%s()->%s() failed to create %s . Exception=%s", objName, __func__, "ts2", exc.what());
			printf("%s \n", dbg->errmsg);
			throw std::exception(dbg->errmsg);
		}


		success();
	}

	void dump() {

	}
};

struct sChildKaz : sLbase {
	int ChildKazProp;

	sChildKaz(char* kazName_, int ChildKazProp_, bool forceFail, sDebuger* dbg_=nullptr) : sLbase(this, kazName_, dbg_) {
		ChildKazProp=ChildKazProp_;
		
		start("ChildKazProp=%d", ChildKazProp);
		//-- ... do specific constructor stuff ...

		if (forceFail) {
			fail("forceFail set. ChildKazProp=%d", ChildKazProp);
		}

		success();
	}

	void childMethod(bool fail_) {
		start("fail_=%s", ((fail_)?"true":"false"));
		if (fail_) {
			fail("forceFail set. fail_=%d", fail_);
		}

		success();
	}

};
struct sParentKaz : sLbase {
	int forecasterProp;

	sChildKaz* child1=nullptr;
	sChildKaz* child2=nullptr;

	sParentKaz(char* kazName_, int forecasterProp_, bool forceFail=false, sDebuger* dbg_=nullptr) : sLbase(this, kazName_, dbg_) {
		forecasterProp=forecasterProp_;
		
		start("forecasterProp=%d", forecasterProp);
		//-- ... do specific constructor stuff ...


		//-- Child Creation SUCCESS
		newC(child1, new sChildKaz("child1 Name", 1, false) );

		//-- Child Method SUCCESS
		callM(child1->childMethod(false));
		
		//-- Child Creation FAILURE
		//newC(child2, new sChildKaz("child2 Name", 2, true));

		try {
			child2= new sChildKaz("child2 Name", 2, true);
				subObj[subObjCnt]=child2;
				subObjCnt++;
		}
		catch (std::exception exc) {
			sprintf_s(dbg->errmsg, DBG_ERRMSG_SIZE, "%s()->%s() failed to create %s . Exception=%s", objName, __func__, "Child2", exc.what()); 
			throw std::exception(dbg->errmsg);
		}



		//-- Internal failure
		if (forceFail) {
			fail("forceFail set. forecasterProp=%d", forecasterProp);
		}

	}
};
*/

struct sDio: sLbase {

	sDio(void* parentObj_, char* objName_, int parm1, int parm2, bool fail_) : sLbase(parentObj_, objName_) {
		start("dio parms: %d , %d", parm1, parm2);

		if (fail_) failC("blah blah");

		success();
	}
};

struct sRoot : sLbase {
	int kaz;

	sRoot(sDebuger* rootdbg_, bool rootverbose_):sLbase(rootdbg_, rootverbose_) {
		dbg->verbose=rootverbose_;
	}

	void go() {

		//======= ALL MAIN PROGRAM GOES HERE !!! =========

		sDio* dio12=nullptr;
		sDio* dio34=nullptr;

		start("action start message. var1=%d , var2=%d", 1, 2);

		try {

			try {
				nakedNew(dio12, sDio, 1, 2, false);
				subObj[subObjCnt]=dio12;
				subObjCnt++;
			}
			catch (std::exception exc) {
				//sprintf_s(dbg->stackmsg, DBG_ERRMSG_SIZE, "%s\n\t%s()->%s() failed to create %s . Exception=%s", dbg->stackmsg, objName, __func__, "dio12", exc.what());
				cleanup(dbg->stackmsg);
				throw std::exception(dbg->stackmsg);
			}

			
			msgbld("Trying: %s = new %s(%s) ...", dio34, sDio, dbg->msg);
			try {
				nakedNew(dio34, sDio, 3, 4, true);
				subObj[subObjCnt]=dio34;
				subObjCnt++;
			}
			catch (std::exception exc) {
				//sprintf_s(dbg->stackmsg, DBG_ERRMSG_SIZE, "%s\n\t%s()->%s() failed to create %s . Exception=%s", dbg->stackmsg, objName, __func__, "dio34", exc.what());
				cleanup(dbg->stackmsg);
				throw std::exception(dbg->stackmsg);
			}

			success();
		}
		catch (std::exception exc) {
			msgbld("Root object failed. k=%d, j=%d", -3, -4);
			dbg->out();
		}

	}
};

void mainCleanup(sDebuger* dbg, int objCnt, ...) {
	va_list args;
	sLbase* obj;

	va_start(args, objCnt);
	for (int o=0; o<objCnt; o++) {
		obj=va_arg(args, sLbase*);
		//-- append child object's dbg->stackmsg to main()'s before exit
		//sprintf_s(dbg->stackmsg, DBG_ERRMSG_SIZE, "%s\n\t%s", dbg->stackmsg, obj->dbg->stackmsg);
		delete (obj);
	}
	va_end(args);
}
int main() {
	char objName[MAX_PATH]="mainClient";
	int	 stackLevel=0;

	sDebuger* dbg=nullptr;


	//-- main debugger
	try {
		dbg=new sDebuger("mainDebugger.log", false);
	}
	catch (std::exception exc) {
		mainFail("Could not create mainDebugger.\n");
	}

	//-- root object
	try {
		sRoot* root=new sRoot(dbg, true);
		//-- main actions are moved into root->go() method, which ca
		root->go();
	}
	catch (std::exception exc) {
		dbg->out();
	}
	dbg->out(true);

	system("pause");

}