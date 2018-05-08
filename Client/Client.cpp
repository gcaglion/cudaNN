#include "../CommonEnv.h"
/*#include "../ParamMgr/ParamMgr.h"
#include "../Data/Data.h"
#include "../Engine/Engine.h"
#include "../Logger/Logger.h"

struct sRoot : s0 {

	sRoot(int argc=0, char* argv[]=nullptr, sDebuggerParms* rootdbgparms_=nullptr) : s0("root", nullptr, rootdbgparms_) {


		//tParmsSource* xparms=new tParmsSource("xparms", this, "C:\\Users\\gcaglion\\dev\\cudaNN\\Client\\Client.xml", argc, argv, true);


		//-- a) declarations for all objects used throughout the client
		
		tParmsSource*	XMLparms=nullptr;	//-- Forecaster XML parameters
		tData*			fData=nullptr;		//-- Forecaster data
		tEngine*		fEngine=nullptr;	//-- Forecaster engine
		tLogger*		fPersistor=nullptr;	//-- Forecaster Persistor

		//-- b) do everything here
		try {

			//-- 1. create client parms, include command-line parms, read parameters file, and parse it on creation
			safespawn(XMLparms, tParmsSource, "C:\\Users\\gcaglion\\dev\\cudaNN\\Client\\Client.xml", argc, argv, true);

			

			//-- 2. create Forecaster Data from parms
			safespawn(fData, tData, XMLparms, ".Forecaster.Data");
			//-- 3. create Forecaster Engine from parms
			safespawn(fEngine, tEngine, XMLparms, ".Forecaster.Engine", fData->shape);
			//-- 4. create Forecaster Persistor
			safespawn(fPersistor, tLogger, XMLparms, ".Forecaster.Persistor");

			int kaz=0;

		}
		catch (std::exception exc) {
			throw(exc);
		}
	}

};
*/

//-- client closures
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


#include "../s0/s0.h"
#include "../ParamMgr/ParamMgr.h"
#include "../Data/Data.h"
#include "../Engine/Engine.h"
#include "../Logger/Logger.h"


struct sRoot : s0 {

	sRoot(int argc=0, char* argv[]=nullptr, sDebuggerParms* rootdbgparms_=nullptr) : s0("root", nullptr, rootdbgparms_) {

		//-- 1. declarations
		tParmsSource*	xparms=nullptr;
		tData*			fData=nullptr;		//-- Forecaster data
		tEngine*		fEngine=nullptr;	//-- Forecaster engine
		tLogger*		fPersistor=nullptr;	//-- Forecaster Persistor

		//-- 2. do stuff
		safespawn(xparms, tParmsSource, "C:\\Users\\gcaglion\\dev\\cudaNN\\Client\\Client.xml", argc, argv, true);
		//-- 2. create Forecaster Data from parms
		safespawn(fData, tData, XMLparms, ".Forecaster.Data");
		//-- 3. create Forecaster Engine from parms
		safespawn(fEngine, tEngine, XMLparms, ".Forecaster.Engine", fData->shape);
		//-- 4. create Forecaster Persistor
		safespawn(fPersistor, tLogger, XMLparms, ".Forecaster.Persistor");

		try {
			xparms=new tParmsSource("xparms", this, "C:\\Users\\gcaglion\\dev\\cudaNN\\Client\\Client.xml", argc, argv, true);
		}
		catch (std::exception exc) {
			fail("Could not create debugger outfile. Exception: %s", exc.what());
		}
	}

};

int main(int argc, char* argv[]) {
	
	//-- 1. create root object. root constructor does everything else
	sRoot* root=nullptr;
	try {
		root=new sRoot();	//-- always takes default debugger settings
	}
	catch (std::exception exc) {
		clientFail("Exception thrown by root. See stack.");
	}

	clientSuccess();
}
