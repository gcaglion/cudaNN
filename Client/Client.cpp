#include "../CommonEnv.h"
#include "../Forecaster/Forecaster.h"

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

void kaz() {
#define shortLen 12
#define longLen 20
	char shorts[shortLen+1];
	char longs[longLen+1];

	shorts[shortLen]='\0';
	longs[longLen]='\0';
	//memcpy_s(shorts, shortLen, "012345678901", shortLen);
	strcpy_s(shorts, longLen, "012345678901");
	strcpy_s(longs, longLen, "01234567890123456789");

	//-- 1. long into short
	strcpy_s(shorts, shortLen, longs);
	strcpy_s(shorts, longLen, longs);
	//-- 2. short into long
	strcpy_s(longs, longLen, shorts);
	strcpy_s(longs, shortLen, shorts);
}
int main(int argc, char* argv[]) {

	//kaz();
	//system("pause");
	//return -1;

	//-- start client timer
	DWORD mainStart=timeGetTime();

	//-- main debugger declaration & creation
	createMainDebugger(DBG_LEVEL_STD, DBG_DEST_BOTH);
	//-- set main debugger properties
	dbg->timing=false;

	tParmsSource* XMLparms=nullptr;
	tForecaster* forecaster=nullptr;
	
	//-- everything else must be enclosed in try/catch block
	try {

		//-- create client parms, include command-line parms, and read parameters file
		safeCallEE(XMLparms=new tParmsSource("C:\\Users\\gcaglion\\dev\\cudaNN\\Client\\Client.xml", argc, argv));

		safeCallEE(XMLparms->parse());

		//-- create Data Forecaster from parms
		safeCallEE(forecaster=new tForecaster(XMLparms, "Forecaster"));
}
	catch (std::exception e) {
		dbg->write(DBG_LEVEL_ERR, "\nClient failed with exception: %s\n", 1, e.what());
		Cleanup( 3, forecaster, XMLparms, dbg);
		return -1;
	}

	Cleanup(3, forecaster, XMLparms, dbg);
	system("pause");
	return 0;

}
