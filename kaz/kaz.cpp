#include "../CommonEnv00.h"
#include "../FileInfo/FileInfo.h"

#include "kazM.h"

#define DBG_DEFAULT_VERBOSE false
#define DBG_ERRMSG_SIZE 32768
#define MAX_OBJ_CHILDREN 32

struct sDebuger {
	int stackLevel;
	bool verbose;
	tFileInfo* outfile;
	char errmsg[DBG_ERRMSG_SIZE]="";

	sDebuger(char* outfilename, bool verbose_=false) {
		verbose=verbose_;
		outfile=new tFileInfo(outfilename, "C:/temp", FILE_MODE_WRITE);
	}
	~sDebuger() {
		printf("%s() called on %p. Deleting outFile (%s) ...\n", __func__, this, outfile->FullName);
		delete outfile;
	}
};

struct sLbase {
	char objName[MAX_PATH]="";
	char errtmp[DBG_ERRMSG_SIZE]="";

	int	 stackLevel=0;
	void*	parentObj;
	int		subObjCnt=0;
	void**	subObj;

	sDebuger* dbg;

	void sLbase_common(void* parentObj_, char* objName_, bool verbose_, sDebuger* dbg_) {
		parentObj=parentObj_;
		stackLevel=((sLbase*)parentObj)->stackLevel+1;
		sprintf_s(objName, MAX_PATH, "%s(%p)", objName_, this);
		subObj=(void**)malloc(MAX_OBJ_CHILDREN*sizeof(void*));

		//-- if we didn't pass a valid dbg_, regardless of what this new object does, it will always create its own debugger
		if (dbg_==nullptr) {
			try {
				dbg=new sDebuger(objName, verbose_);
				dbg->stackLevel=stackLevel+1;
				info("new sDebugger(%s) completed successfully. dbg->stackLevel=%d", objName, dbg->stackLevel);
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

	sLbase(void* parentObj_, char* objName_, bool verbose_) {
		sLbase_common(parentObj_, objName_, verbose_, nullptr);
	}

	sLbase(void* parentObj_, char* objName_, sDebuger* dbg_) {
		sLbase_common(parentObj_, objName_, DBG_DEFAULT_VERBOSE, dbg_);
	}

	void cleanup(char* failmsg) {

		info("%s->cleanup() called; failmsg=%s", objName, failmsg);
		for (int o=0; o<subObjCnt; o++) {
			info("%s->cleanup() calling %s->cleanup() ...", objName, ((sLbase*)subObj[o])->objName);
			((sLbase*)subObj[o])->cleanup("called from parent cleanup()");
		}
		info("%s->cleanup() calling (delete dbg) ...", objName);
		delete dbg;
	}

	~sLbase() {
		info("%s->%s() called; calling cleanup()", objName, __func__);
		cleanup("destructor call");
	}
};

struct sChildKaz : sLbase {
	int ChildKazProp;

	sChildKaz(char* kazName_, int ChildKazProp_, bool forceFail, sDebuger* dbg_=nullptr) : sLbase(this, kazName_, dbg_) {
		ChildKazProp=ChildKazProp_;
		
		info("%s->%s() called. ChildKazProp=%d", objName, __func__, ChildKazProp);
		//-- ... do specific constructor stuff ...

		if (forceFail) {
			failM("forceFail set. ChildKazProp=%d", ChildKazProp);
		}

		info("%s->%s() successful.", objName, __func__);
	}

	void childMethod(bool fail_) {
		info("%s->%s() called. fail_=%s", objName, __func__, ((fail_)?"true":"false"));
		if (fail_) {
			err("%s->%s() failed, because of fail_", objName, __func__);
			throw std::exception(dbg->errmsg);
		}
		info("%s->%s() successful.", objName, __func__);
	}

};

struct sParentKaz : sLbase {
	int forecasterProp;

	sChildKaz* child1=nullptr;
	sChildKaz* child2=nullptr;

	sParentKaz(char* kazName_, int forecasterProp_, bool forceFail=false, sDebuger* dbg_=nullptr) : sLbase(this, kazName_, dbg_) {
		forecasterProp=forecasterProp_;
		
		//-- ... do specific constructor stuff ...


		//-- Child Creation SUCCESS
		spawn(child1, new sChildKaz("child1 Name", 1, false) );

		//-- Child Method SUCCESS
		method(child1->childMethod(false));
		
		//-- Child Creation FAILURE
		//spawn(child2, new sChildKaz("child2 Name", 2, true));
		try {
				child2 = new sChildKaz("child2 Name", 2, true);
				subObj[subObjCnt]=child2; 
				subObjCnt++; 
		}
		catch (std::exception exc) {
			//-- put all this into dbg->write()
			sprintf_s(dbg->errmsg, DBG_ERRMSG_SIZE, "%s()->%s() failed to create %s . Exception=%s", objName, __func__, "child2", exc.what()); 
			printf("%s\n", dbg->errmsg);
			fprintf(dbg->outfile->handle, "%s\n", dbg->errmsg);
			//--
			throw std::exception(dbg->errmsg);
		} 


		//-- Internal failure
		if (forceFail) {
			failM("forceFail set. forecasterProp=%d", forecasterProp);
		}

	}
};

void mainCleanup(int objCnt, ...) {
	va_list args;
	sLbase* obj;

	va_start(args, objCnt);
	for (int o=0; o<objCnt; o++) {
		obj=va_arg(args, sLbase*);
		delete (obj);
	}
	va_end(args);
}

int main() {

	sDebuger* dbg=nullptr;
	sParentKaz* forecaster=nullptr;
	sChildKaz* timeserie=nullptr;

	//-- main debugger
	try {
		dbg=new sDebuger("mainDebugger", true);
	}
	catch (std::exception exc) {
		mainFail("Could not create mainDebugger.\n");
	}

	//-- main actions

	try {
		forecaster=new sParentKaz("Forecaster", 0, false);
	}
	catch (std::exception exc) {
		mainCleanup(2, timeserie, forecaster);
		mainFail("forecaster creation failed. Exception=%s \n", exc.what());
	}

	mainCleanup(2, timeserie, forecaster);
	mainSuccess();

}