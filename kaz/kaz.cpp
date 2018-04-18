#include "../CommonEnv.h"
#include "../FileInfo/FileInfo.h"
#include "../ParamMgr/ParamMgr.h"
#include <memory>


void Cleanup(int objCnt, ...) {
	va_list args;
	sBaseObj* obj;

	va_start(args, objCnt);
	for (int o=0; o<objCnt; o++) {
		obj=va_arg(args, sBaseObj*);
		delete (obj);
	}
	va_end(args);
}
/*truct sElement {
	int id;
	int type;

	char* errstack;
	char* errmsg;

	sElement(int id_, int type_) { 
		errstack=(char*)malloc(1024);
		id=id_; type=type_; 
		if (id==2) {
			sprintf_s(errstack, 1024, "sElement() failed to create element %d!\n", id);
			throw(errstack);
		}
	}

	~sElement(){
		printf("sElement() destructor called.\n");
		free(errstack);
	}

};
struct sContainer {
	int id;
	int type;

	char* errstack;

	sElement* elem[4];

	sContainer() {
		errstack=(char*)malloc(1024);
		for (int e=0; e<3; e++) {
			try {
				elem[e]=new sElement(e, e);
			}
			catch (char* exc) {
				sprintf_s(errstack, 1024, "sContainer() failed to add elem[%d]! Underlying exception: %s\n", e, exc);
				cleanup(e);
				throw(errstack);
			}
		}
	}

	void cleanup(int upto) {
		for (int e=0; e<upto; e++) {
			delete elem[e];
		}
	}
	~sContainer() {
		printf("sContainer() destructor called.\n");
		for (int e=0; e<3; e++) {
			delete elem[e];
		}
		free(errstack);
	}
};
*/

#define DBG_ERRMSG_SIZE 32768

struct sLbase {
	bool verbose=true;
	char objName[MAX_PATH]="";
	char errstack[DBG_ERRMSG_SIZE]="";
	char errmsg[DBG_ERRMSG_SIZE]="";

	sLbase(char* objName_) {
		sprintf_s(objName, MAX_PATH, "%s(%p)", objName_, this);
	}
	//virtual void cleanup(char* msg) { printf("sLbase->cleanup() called with msg: %s.\n", msg); }
};

void compose(char* msg_, char* omsg, int argcount, ...) {
	va_list			arguments;
	char submsg[MAX_PATH];
	char*			arg_s;
	int				arg_d;
	double			arg_f;
	unsigned int	im = 0;
	int				prev_im = 0;
	char tmpmsg[DBG_ERRMSG_SIZE];
	char msg[DBG_ERRMSG_SIZE];

	va_start(arguments, argcount);
	removeQuotes(msg_, msg);
	omsg[0]='\0';
	do {
		if (msg[im]==37) {                // "%"
			memcpy(submsg, &msg[prev_im], (im-prev_im+2)); submsg[im-prev_im+2] = '\0';
			prev_im = im+2;
			if (msg[im+1]==115) {   // "s"
				arg_s = va_arg(arguments, char*);
				sprintf_s(tmpmsg, submsg, arg_s);
				strcat_s(omsg, DBG_ERRMSG_SIZE, tmpmsg);
			} else if (submsg[im+1]==100) {   // "d"
				arg_d = va_arg(arguments, int);
				sprintf_s(tmpmsg, submsg, arg_d); strcat_s(omsg, DBG_ERRMSG_SIZE, tmpmsg);
			} else if (submsg[im+1]==112) {   // "p"
				arg_d = va_arg(arguments, long);
				sprintf_s(tmpmsg, submsg, arg_d); strcat_s(omsg, DBG_ERRMSG_SIZE, tmpmsg);
			} else {   // this could be 67 ("f") or any mask before "f" -> in any case, it's a double
				arg_f = va_arg(arguments, double);
				sprintf_s(tmpmsg, submsg, arg_f); strcat_s(omsg, DBG_ERRMSG_SIZE, tmpmsg);
			}
		}
		im++;
	} while (im<strlen(msg));

	memcpy(submsg, &msg[prev_im], (im-prev_im+2)); submsg[im-prev_im+2] = '\0';
	sprintf_s(tmpmsg, submsg, arg_s);
	strcat_s(omsg, DBG_ERRMSG_SIZE, tmpmsg);

	va_end(arguments);
}
void removeQuotes(char* istr, char* ostr) {
	size_t slen=strlen(istr);
	size_t rlen=slen;
	int ri=0;
	for (int si=0; si<slen; si++) {
		if (istr[si]!=34) {
			ostr[ri]=istr[si];
			ri++;
		}
	}
	ostr[ri]='\0';
}

/*#define fail(failMsgMask, argcnt, ...){ \
	compose((#failMsgMask), errmsg, argcnt, __VA_ARGS__ ); \
	sprintf_s(errstack, DBG_ERRMSG_SIZE, "%s() -> %s() failed with message: %s\n", errstack, __func__, errmsg); \
	printf("%s->fail() macro called compose(); \n\t errmsg : %s \n\t errstack : %s\n", objName, errmsg, errstack); \
	cleanup("FAIL MACRO"); \
	throw std::exception(errstack); \
}
*/
#define fail(failMsgMask, argcnt, ...){ \
	compose((#failMsgMask), errmsg, argcnt, __VA_ARGS__ ); \
	sprintf_s(errstack, DBG_ERRMSG_SIZE, "%s", errmsg); \
	if(verbose) printf("%s\n", errstack); \
	cleanup("FAIL MACRO"); \
	throw std::exception(errstack); \
}

#define cmdlow(cmd){ \
	sprintf_s(errstack, DBG_ERRMSG_SIZE, "%s \n || %s()->%s() calling %s...", errstack, objName, __func__, #cmd); \
	if(verbose) printf("%s\n", errstack); \
	try { \
		cmd; \
		sprintf_s(errstack, DBG_ERRMSG_SIZE, "%s \n || %s()->%s() call to %s SUCCESS...", errstack, objName, __func__, #cmd); \
		if(verbose) printf("%s\n", errstack); \
	} \
	catch (std::exception exc) { \
		sprintf_s(errstack, DBG_ERRMSG_SIZE, "%s \n || %s()->%s() call to %s FAILED...", errstack, objName, __func__, #cmd); \
		if(verbose) printf("%s\n", errstack); \
		cleanup("CMDLOW MACRO"); \
		throw std::exception(errstack); \
	} \
}
#define bottomsuccess() { \
		sprintf_s(errstack, DBG_ERRMSG_SIZE, "%s \n || %s()->%s() call (Lowest Level) SUCCESS...", errstack, objName, __func__); \
		if(verbose) printf("%s\n", errstack); \
}

struct sL3 : sLbase {
	int L3id;

	sL3(bool L3fail) :sLbase("L3parm") {
		if (L3fail) {
			fail("parameter value=%d", 1, 3);
		} else {
			bottomsuccess();
		}
	}

	~sL3() {
		printf("sL3 destructor called.\n");
	}

	void cleanup(char* msg) {
		printf("%s->cleanup() called with msg=%s.\n", objName, msg);
		//-- delete all objects (potentially) created by constructor before its failure
	}

};
struct sL2 : sLbase {
	int L2id;

	sL3* l3=nullptr;

	sL2(bool L2fail, bool L3fail) :sLbase("L2parm") {
		if (L2fail) {
			fail("parameter value=%d", 1, 2);
		} else {
			cmdlow(l3=new sL3(L3fail));
		}
	}

	~sL2() {
		printf("sL2 destructor called.\n");
	}

	void cleanup(char* msg) {
		printf("%s->cleanup() called with msg=%s.\n", objName, msg);
		//-- delete all objects (potentially) created by constructor before its failure
		delete l3;
	}

};
struct sL1 : sLbase {
	int L1id;

	sL2* l2good=nullptr;
	sL2* l2bad=nullptr;
	sL2* l2=nullptr;

	sL3* l3bad=nullptr;

	sL1(bool L1fail, bool L2fail, bool L3fail) :sLbase("L1parm") {
		if (L1fail) {
			cmdlow(l3bad=new sL3(false));
			fail("parameter value=%d", 1, 1);
		} else {
			//cmdlow(successMethod());
			//cmdlow(l2good=new sL2(false, false));
			//cmdlow(l2bad=new sL2(true, true));
			cmdlow(l2=new sL2(L2fail, L3fail));
		}
	}
	sL1() :sLbase("L1parm") {
		cmdlow(successMethod());
	}

	~sL1() {
		printf("%s() destructor called.\n", objName);
		cleanup("called by destructor.");
	}

	void failMethod() {
		fail("parameter value=%d", 1, -1);
	}
	void successMethod() {
		bottomsuccess();
	}

	void cleanup(char* msg) {
		printf("%s->cleanup() called with msg=%s.\n", objName, msg);
		//-- delete all objects (potentially) created by constructor before its failure
		delete l2good;
		delete l2bad;
	}

};

struct sL0 : sLbase {
	int L0id;

	//-- objects created within this
	sL3* l3good=nullptr;
	sL1* l1good=nullptr;
	sL1* l1bad=nullptr;

	sL0(bool L0fail, bool L1fail, bool L2fail, bool L3fail) :sLbase("L0parm") {
		if (L0fail) {
			fail("parameter value=%d", 1, 0);
		} else {
			cmdlow(l1good=new sL1(true, L2fail, L3fail));
		}
	}
	sL0() :sLbase("L0parm") {
		//cmdlow(l3good=new sL3(false));
		cmdlow(l1good=new sL1(false, false, false));
		cmdlow(l1bad=new sL1(true, false, false));
	}

	~sL0() {
		printf("sL0 destructor called.\n");
	}

	void cleanup(char* msg) { 
		printf("%s->cleanup() called with msg=%s.\n", objName, msg); 
		//-- delete all objects (potentially) created by constructor before its failure
		delete l3good;
		delete l1good;
		delete l1bad;
	}

};

//-- this is used by main(), and deletes all objects (potentially) created by main() before its failure
#define cleanup(msg){ \
	printf("macro cleanup() called with msg=%s.\n", msg); \
	delete l0; \
	printf("aaa\n"); \
	delete l1; \
}

int main() {
/*
	char errstack[DBG_ERRMSG_SIZE]="";
	char parentObjName[MAX_PATH]="";
	char* exc="Exception text";
	sprintf_s(errstack, DBG_ERRMSG_SIZE, "%s -> %s() FAILURE: %s\n", parentObjName, __func__, exc);
	printf("%s\n", errstack);
*/

	char* objName="main";
	char errstack[DBG_ERRMSG_SIZE]="";
	bool  verbose=true;

	sL0* l0=nullptr;
	sL1* l1=nullptr;

	try {
		//cmdlow(l0=new sL0(false, false, false, false));

		cmdlow(l1=new sL1());
		cmdlow(l0=new sL0());
	}
	catch (std::exception exc) {
		printf("\n\n---------------------------------------------------------------------------------\nFull stack:\n%s\n", exc.what());
	}


	delete l1;
	delete l0;
	cleanup("DioPorco!");

	system("pause");
	return -1;
/*
	sContainer* cont=nullptr;
	try { 
		cont=new sContainer(); 
	}
	catch (char* exc) {
		printf("Client failed. Exception: %s\n", exc);
		delete cont;
	}

	return -1;

	tParmsSource* p0=nullptr;
	tDebugger* dbg1=nullptr;
	tParmsSource* p1=nullptr;

	try {
		p0= new tParmsSource("C:\\Users\\gcaglion\\dev\\cudaNN\\Client\\Client.xml", 0, NULL, nullptr);
		dbg1=new tDebugger(DBG_LEVEL_ERR, DBG_DEST_BOTH, "dbg1.log");
		p1= new tParmsSource("C:/temp/parms1.xml", 0, NULL, dbg1);
	}
	catch (char* e) {
		printf("Client Failed! Exception: %s\n", e);
		Cleanup(3, p0, dbg1, p1);
		system("pause");
		return -1;
	}
*/
/*	tFileInfo* f0=nullptr;
	tFileInfo* f1=nullptr;
	tFileInfo* f2=nullptr;
	tFileInfo* f3=nullptr;
	tFileInfo* f4=nullptr;
	try {
		f0=new tFileInfo("C:/temp/logs/f0.log", FILE_MODE_WRITE);
		f1=new tFileInfo("f1.log", FILE_MODE_WRITE);
		f2=new tFileInfo("f2log", FILE_MODE_WRITE);
		f3=new tFileInfo("C:/temp/logs/f3log", FILE_MODE_WRITE);
		f4=new tFileInfo("C:\\temp\\logs\\f3log", FILE_MODE_WRITE);
	}
	catch (std::exception e) {
		printf("\nClient failed with exception: %s\n", e.what());
		Cleanup( 5, f0, f1, f2, f3, f4);
		system("pause");
		return -1;
	}

	Cleanup( 5, f0, f1, f2, f3, f4);
*/

	
	printf("Client Success!\n");
	//Cleanup(3, p0, dbg1, p1);
	system("pause");
	return 0;
}