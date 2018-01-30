//#include <vld.h>
#include "TimeSerie.h"

int sTS::LoadOHLCVdata(char* date0) {

	if (OraConnect(DebugParms, FXData->FXDB)!=0) return -1;
	if (Ora_GetFlatOHLCV(DebugParms, FXData->FXDB->DBCtx, FXData->Symbol, FXData->TimeFrame, date0, this->steps, this->hd, this->hbd)!=0) return -1;

	return 0;
}


void sTS::SBF2BFS(int db, int ds, int dbar, int df, numtype* iSBFv, numtype* oBFSv) {
	int i=0;
	for (int b=0; b<db; b++) {
		for (int bar=0; bar<dbar; bar++) {
			for (int f=0; f<df; f++) {
				for (int s=0; s<ds; s++) {
					oBFSv[i]=iSBFv[b* ds*dbar*df+s*dbar*df+bar*df+f];
					i++;
				}
			}
		}
	}
}
void sTS::BFS2SBF(int db, int ds, int dbar, int df, numtype* iBFSv, numtype* oSBFv) {
	int i=0;
	for (int b=0; b<db; b++) {
		for (int s=0; s<ds; s++) {
			for (int bar=0; bar<dbar; bar++) {
				for (int f=0; f<df; f++) {
					oSBFv[i]=iBFSv[b* dbar*df*ds+bar*df*ds+f*ds+s];
					i++;
				}
			}
		}
	}
}

int sTS::load(tFXData* tsFXData_, char* pDate0) {
	FXData=tsFXData_;
	sourceType=SOURCE_DATA_FROM_FXDB;
	return (LoadOHLCVdata(pDate0));
}
int sTS::load(tFileData* tsFileData, char* pDate0) {
	return -1;
}
int sTS::load(tMT4Data* tsMT4Data, char* pDate0) {
	return -1;
}
