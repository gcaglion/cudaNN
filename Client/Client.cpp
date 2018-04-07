#include "../CommonEnv.h"
#include "../Debugger/Debugger.h"
#include "../Forecaster/Forecaster.h"

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

	//-- everything else must be enclosed in try/catch block
	try {

		//-- create client parms, include command-line parms, and read parameters file
		tParmsSource* XMLparms; safeCallEE(XMLparms=new tParmsSource("C:\\Users\\gcaglion\\dev\\cudaNN\\Client\\Client.xml", argc, argv));

		XMLparms->parse();

		//-- create Data Forecaster from parms
		tForecaster* forecaster; safeCallEE(forecaster=new tForecaster(XMLparms, "Forecaster", dbg));
}
	catch (std::exception e) {
		dbg->write(DBG_LEVEL_ERR, "\nClient failed with exception: %s\n", 1, e.what());
		return -1;
	}

	system("pause");
	return 0;

}
