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

	sDebuger(char* outfilename, bool verbose_=false) {
		verbose=verbose_;
		outfile=new tFileInfo(outfilename, "C:/temp", FILE_MODE_WRITE);
	}
	~sDebuger() {
		//for (int t=0; t<stackLevel; t++) printf("\t");
		printf("%s() called on %p. Deleting outFile (%s) ...\n", __func__, this, outfile->FullName);
		delete outfile;
	}
};

struct sLbase {
	char objName[MAX_PATH]="";
	char errmsg[DBG_ERRMSG_SIZE]="";
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
	~sChildKaz() {
		info("%s->%s() called. ChildKazProp=%d", objName, __func__, ChildKazProp);
	}

	void childMethod(bool fail_) {
		info("%s->%s() called. fail_=%s", objName, __func__, ((fail_)?"true":"false"));
		if (fail_) {
			err("%s->%s() failed, because of fail_", objName, __func__);
			throw std::exception(errmsg);
		}
		info("%s->%s() successful.", objName, __func__);
	}

};

struct sParentKaz : sLbase {
	int ParentKazProp;

	sChildKaz* child1=nullptr;
	sChildKaz* child2=nullptr;

	sParentKaz(char* kazName_, int ParentKazProp_, bool forceFail=false, sDebuger* dbg_=nullptr) : sLbase(this, kazName_, dbg_) {
		ParentKazProp=ParentKazProp_;
		
		//-- ... do specific constructor stuff ...

		//-- Child Creation SUCCESS
		spawn(child1, new sChildKaz("child1 Name", 1, false) );

		//-- Child Method SUCCESS
		method(child1->childMethod(false));
		
		//-- Child Creation FAILURE
		spawn(child2, new sChildKaz("child2 Name", 2, true));

		//-- Internal failure
		if (forceFail) {
			sprintf_s(errmsg, DBG_ERRMSG_SIZE, "%s()->%s() failed in itself. Problem=%s", objName, __func__, "Internal cause of failure");
			cleanup(errmsg);
		}

		sprintf_s(errmsg, "Parent %s constructor failed", __func__);
		throw std::exception(errmsg);


	}
	~sParentKaz() {
		info("ParentKaz destructor called. ParentKazProp=%d", ParentKazProp);
	}
};

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
		forecaster=new sParentKaz("Forecaster", 0);
	}
	catch (std::exception exc) {
		mainFail("forecaster creation failed. Exception=%s \n", exc.what());
	}

	mainSuccess();

}