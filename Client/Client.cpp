#include "../CommonEnv.h"
#include "../ParamMgr/ParamMgr.h"
#include "../Data/Data.h"
#include "../Engine/Engine.h"
#include "../Logger/Logger.h"

struct sRoot : sBaseObj {

	sRoot(int argc=0, char* argv[]=nullptr, sDebuggerParms* rootdbgparms_=nullptr) : sBaseObj("root", nullptr, rootdbgparms_) {

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
