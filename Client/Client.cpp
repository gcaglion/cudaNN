#include "../CommonEnv.h"
#include "../Forecaster/Forecaster.h"

#define SUCCESS 0
#define FAILURE -1
#define Bye(retcode) { system("pause"); return retcode; }

void objCleanup(int objCnt, ...) {
	va_list args;
	sBaseObj* obj;

	va_start(args, objCnt);
	for (int o=0; o<objCnt; o++) {
		obj=va_arg(args, sBaseObj*);
		delete (obj);
	}
	va_end(args);
}

void cleanup() {
	printf("\nClient->cleanup() called.\n");
}
int main(int argc, char* argv[]) {

	//-- 1. main debugger declaration & creation. Also declares and starts timer (mainStart)
	createMainDebugger(DBG_LEVEL_STD, DBG_DEST_BOTH);

	//-- 2. objects used throughout the client
	tParmsSource* XMLparms=nullptr;
	tForecaster* forecaster=nullptr;
	tData* data0=nullptr;
	tEngine* engine0=nullptr;

	//==================  main stuff ========================================================================================================
	try {
		//-- create client parms, include command-line parms, and read parameters file
		safeCall(XMLparms=new tParmsSource("C:\\Users\\gcaglion\\dev\\cudaNN\\Client\\Client.xml", argc, argv));
		safeCall(XMLparms->parse());

		//-- create Data Forecaster from parms
		//safeCall(forecaster=new tForecaster(XMLparms, "Forecaster"));

		safeCall(data0=new tData(XMLparms, ".Forecaster.Data"));
		safeCall(engine0=new tEngine(XMLparms, ".Forecaster.Engine", data0->shape));
		delete engine0;

	}
	catch (char* e) {
		objCleanup(4, XMLparms, forecaster, data0, engine0);
		dbg->write(DBG_LEVEL_ERR, "main() failed. Exception=%s\n", 1, e);
		Bye(FAILURE);
	}
	//==========================================================================================================================


	objCleanup(4, XMLparms, forecaster, data0, engine0);
	Bye(SUCCESS);
}
