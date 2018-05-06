#include "../CommonEnv.h"
#include "../Forecaster/Forecaster.h"

struct sRoot : sBaseObj {

	//-- here we put everything that needs to be done
	sRoot(int argc=0, char* argv[]=nullptr, sDebuggerParms* rootdbgparms_=nullptr) : sBaseObj("root", nullptr, rootdbgparms_) {

		//-- 2. objects used throughout the client
		tParmsSource* XMLparms=nullptr;
		tForecaster* forecaster=nullptr;
		//tData* data0=nullptr;
		//tEngine* engine0=nullptr;

		try {

			//-- create client parms, include command-line parms, and read parameters file
			safespawn(XMLparms, tParmsSource, "C:\\Users\\gcaglion\\dev\\cudaNN\\Client\\Client.xml", argc, argv);
			safecall(XMLparms->parse());

			//-- create Data Forecaster from parms
			safespawn(forecaster, tForecaster, XMLparms, "Forecaster");

			//safespawn(data0, tData, XMLparms, ".Forecaster.Data");
			//safespawn(engine0, tEngine, XMLparms, ".Forecaster.Engine", data0->shape);

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
