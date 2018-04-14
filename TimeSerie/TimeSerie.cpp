//#include <vld.h>
#include "TimeSerie.h"

//-- sTimeSerie, constructors / destructor
void sTimeSerie::sTimeSeriecommon(int steps_, int featuresCnt_, int tsfCnt_, int* tsf_, tDebugger* dbg_) {

	steps=steps_;
	featuresCnt=featuresCnt_;
	len=steps*featuresCnt;
	dmin=(numtype*)malloc(featuresCnt*sizeof(numtype));
	dmax=(numtype*)malloc(featuresCnt*sizeof(numtype));
	for (int f=0; f<featuresCnt; f++) {
		dmin[f]=1e8; dmax[f]=-1e8;
	}
	scaleM=(numtype*)malloc(featuresCnt*sizeof(numtype));
	scaleP=(numtype*)malloc(featuresCnt*sizeof(numtype));
	dtime=(char**)malloc(len*sizeof(char*)); for (int i=0; i<len; i++) dtime[i]=(char*)malloc(12+1);
	bdtime=(char*)malloc(12+1);
	d=(numtype*)malloc(len*sizeof(numtype));
	bd=(numtype*)malloc(featuresCnt*sizeof(numtype));
	d_tr=(numtype*)malloc(len*sizeof(numtype));
	d_trs=(numtype*)malloc(len*sizeof(numtype));

}

//-------- To fix --------------
sTimeSerie::sTimeSerie(int steps_, int featuresCnt_, tDebugger* dbg_) {
	sTimeSeriecommon(steps_, featuresCnt_, 0, nullptr, dbg_);
}
sTimeSerie::sTimeSerie(tFXData* dataSource_, int steps_, char* date0_, int dt_, tDebugger* dbg_){
	//-- 1. create
	sTimeSeriecommon(steps_, FXDATA_FEATURESCNT, 0, nullptr, dbg_);	// no safeCall() because we don't set dbg, here
	//-- 2. load data
	safeCallEE(load(dataSource_, date0_));
	//-- 3. transform
	safeCallEE(transform(dt_));
}
sTimeSerie::sTimeSerie(tFileData* dataSource_, int steps_, int featuresCnt_, char* date0_, int dt_, tDebugger* dbg_){
	featuresCnt=featuresCnt_;
}
/*sTimeSerie::sTimeSerie(tMT4Data* dataSource_, int steps_, char* date0_, int dt_, tDebugger* dbg_){
}
//-------------------------------------------------------------------------------------------------------------------------------------
*/
sTimeSerie::sTimeSerie(tParmsSource* parms, char* parmKey, tDebugger* dbg_){

	tsf=(int*)malloc(MAX_TSF_CNT*sizeof(int));
	date0[DATE_FORMAT_LEN]='\0';

	safeCallEB(parms->setKey(parmKey));

	//-- 0. common parameters
	safeCallEE(parms->get(date0, "Date0"));
	safeCallEE(parms->get(&steps, "HistoryLen"));
	safeCallEE(parms->get(&dt, "DataTransformation"));
	safeCallEE(parms->get(&BWcalc, "BWCalc"));
	safeCallEE(parms->get(&tsf, "StatisticalFeatures", &tsfCnt));

	//-- 1. Find DataSource.Type
	safeCallEB(parms->setKey("DataSource"));
	safeCallEE(parms->get(&sourceType, "Type"));

	//-- 2. create DataSource according to Type
	switch (sourceType) {
	case FXDB_SOURCE:
		safeCallEE(fxData=new tFXData(parms, "FXData"));
		featuresCnt=FXDATA_FEATURESCNT;
		break;
	case FILE_SOURCE:
		safeCallEE(fileData=new tFileData(parms, "FileData"));
		featuresCnt=fileData->featuresCnt;
		break;
/*	case MT4_SOURCE:
		safeCallEE(mt4Data=new tMT4Data(parms));
		featuresCnt=mt4Data->featuresCnt;
		break;
*/	default:
		throwE("invalid DataSource Type: %d", 1, sourceType);
	}

	//-- 3. common stuff (mallocs, ...)
	len=steps*featuresCnt;
	dmin=(numtype*)malloc(featuresCnt*sizeof(numtype));
	dmax=(numtype*)malloc(featuresCnt*sizeof(numtype));
	for (int f=0; f<featuresCnt; f++) {
		dmin[f]=1e8; dmax[f]=-1e8;
	}
	scaleM=(numtype*)malloc(featuresCnt*sizeof(numtype));
	scaleP=(numtype*)malloc(featuresCnt*sizeof(numtype));
	dtime=(char**)malloc(len*sizeof(char*)); for (int i=0; i<len; i++) dtime[i]=(char*)malloc(12+1);
	bdtime=(char*)malloc(12+1);
	d=(numtype*)malloc(len*sizeof(numtype));
	bd=(numtype*)malloc(featuresCnt*sizeof(numtype));
	d_tr=(numtype*)malloc(len*sizeof(numtype));
	d_trs=(numtype*)malloc(len*sizeof(numtype));

}
sTimeSerie::~sTimeSerie() {
	free(d);
	free(bd);
	free(d_trs);
	free(d_tr);
	for (int i=0; i<len; i++) free(dtime[i]);
	free(dtime); free(bdtime);
	free(tsf);

	delete fxData;
	delete fileData;
}

//-- sTimeSerie, other methods
bool sTimeSerie::LoadOHLCVdata(char* date0) {

	if (!OraConnect(dbg, fxData->db)) return false;
	if (!Ora_GetFlatOHLCV(dbg, fxData->db, fxData->Symbol, fxData->TimeFrame, date0, this->steps, this->dtime, this->d, this->bdtime, this->bd)) return false;

	return true;
}
void sTimeSerie::load(tFXData* tsFXData_, char* pDate0) {
	fxData=tsFXData_;
	sourceType=FXDB_SOURCE;
	if (!LoadOHLCVdata(pDate0)) throwE("pDate0=%s", 1, pDate0);
}
void sTimeSerie::load(tFileData* tsFileData, char* pDate0) {
	throwE("", 0);
}
/*void sTimeSerie::load(tMT4Data* tsMT4Data, char* pDate0) {
	throwE("", 0);
}*/
void sTimeSerie::dump(char* dumpFileName) {
	int s, f;
	tDebugger* dumpDbg=new tDebugger(DBG_LEVEL_DET, DBG_DEST_FILE, dumpFileName, DBG_DEFAULT_PATH);
	dumpDbg->write(DBG_LEVEL_DET, "i, datetime", 0);
	for (f=0; f<featuresCnt; f++) dumpDbg->write(DBG_LEVEL_DET, ",F%d_orig,F%d_tr,F%d_trs", 3, f, f, f);
	dumpDbg->write(DBG_LEVEL_DET, "\n%d,%s", 2, -1, bdtime);
	for (f=0; f<featuresCnt; f++) {
		dumpDbg->write(DBG_LEVEL_DET, ",%f", 1, bd[f]);
		for (int ff=0; ff<(featuresCnt-3); ff++) dumpDbg->write(DBG_LEVEL_DET, ",", 0);
	}

	for (s=0; s<steps; s++) {
		dumpDbg->write(DBG_LEVEL_DET, "\n%d, %s", 2, s, dtime[s]);
		for (f=0; f<featuresCnt; f++) {
			dumpDbg->write(DBG_LEVEL_DET, ",%f", 1, d[s*featuresCnt+f]);
			if (hasTR) {
				dumpDbg->write(DBG_LEVEL_DET, ",%f", 1, d_tr[s*featuresCnt+f]);
			} else {
				dumpDbg->write(DBG_LEVEL_DET, ",", 0);
			}
			if (hasTRS) {
				dumpDbg->write(DBG_LEVEL_DET, ",%f", 1, d_trs[s*featuresCnt+f]);
			} else {
				dumpDbg->write(DBG_LEVEL_DET, ",", 0);
			}
		}
	}
	dumpDbg->write(DBG_LEVEL_DET, "\n", 0);

	if (hasTR) {
		dumpDbg->write(DBG_LEVEL_DET, "\ntr-min:", 0);
		for (f=0; f<featuresCnt; f++) dumpDbg->write(DBG_LEVEL_DET, ",,,%f", 1, dmin[f]);
		dumpDbg->write(DBG_LEVEL_DET, "\ntr-max:", 0);
		for (f=0; f<featuresCnt; f++) dumpDbg->write(DBG_LEVEL_DET, ",,,%f", 1, dmax[f]);
		dumpDbg->write(DBG_LEVEL_DET, "\n", 0);
	}
	if (hasTRS) {
		dumpDbg->write(DBG_LEVEL_DET, "\nscaleM:", 0);
		for (f=0; f<featuresCnt; f++) dumpDbg->write(DBG_LEVEL_DET, ",,,%f", 1, scaleM[f]);
		dumpDbg->write(DBG_LEVEL_DET, "\nscaleP:", 0);
		for (f=0; f<featuresCnt; f++) dumpDbg->write(DBG_LEVEL_DET, ",,,%f", 1, scaleP[f]);
		dumpDbg->write(DBG_LEVEL_DET, "\n", 0);

		//dumpDbg->write(DBG_LEVEL_DET, "scaleM:,,%f,,,%f,,,%f,,,%f,,,%f \n", scaleM[0], scaleM[1], scaleM[2], scaleM[3], scaleM[4]);
		//dumpDbg->write(DBG_LEVEL_DET, "scaleP:,,%f,,,%f,,,%f,,,%f,,,%f \n", scaleP[0], scaleP[1], scaleP[2], scaleP[3], scaleP[4]);
	}

	delete dumpDbg;

}
void sTimeSerie::transform(int dt_) {
	dt=dt_;
	for (int s=0; s<steps; s++) {
		for (int f=0; f<featuresCnt; f++) {
			switch (dt) {
			case DT_NONE:
				break;
			case DT_DELTA:
				if (s>0) {
					d_tr[s*featuresCnt+f]=d[s*featuresCnt+f]-d[(s-1)*featuresCnt+f];
				} else {
					d_tr[s*featuresCnt+f]=d[s*featuresCnt+f]-bd[f];
				}
				break;
			case DT_LOG:
				break;
			case DT_DELTALOG:
				break;
			default:
				break;
			}
			if (d_tr[s*featuresCnt+f]<dmin[f]) {
				dmin[f]=d_tr[s*featuresCnt+f];
			}
			if (d_tr[s*featuresCnt+f]>dmax[f]) {
				dmax[f]=d_tr[s*featuresCnt+f];
			}
		}
	}

	hasTR=true;
}
void sTimeSerie::scale(numtype scaleMin_, numtype scaleMax_) {
	//-- ScaleMin/Max depend on the core, scaleM/P are specific for each feature

	scaleMin=scaleMin_; scaleMax=scaleMax_;

	if (!hasTR) throwE("-- must transform before scaling! ---", 0);

	for (int f=0; f<featuresCnt; f++) {
		scaleM[f] = (scaleMax-scaleMin)/(dmax[f]-dmin[f]);
		scaleP[f] = scaleMax-scaleM[f]*dmax[f];
	}

	for (int s=0; s<steps; s++) {
		for (int f=0; f<featuresCnt; f++) {
			d_trs[s*featuresCnt+f]=d_tr[s*featuresCnt+f]*scaleM[f]+scaleP[f];
		}
	}

	hasTRS=true;
}
void sTimeSerie::TrS(int dt_, numtype scaleMin_, numtype scaleMax_) {
	dt=dt_;

	int s, f;
	//-- first, transform
	for (s=0; s<steps; s++) {
		for (f=0; f<featuresCnt; f++) {
			dbg->write(DBG_LEVEL_DET, ",%f", 1, d[s*featuresCnt+f]);
			switch (dt) {
			case DT_NONE:
				break;
			case DT_DELTA:
				if (s>0) {
					d_tr[s*featuresCnt+f]=d[s*featuresCnt+f]-d[(s-1)*featuresCnt+f];
				} else {
					d_tr[s*featuresCnt+f]=d[s*featuresCnt+f]-bd[f];
				}
				break;
			case DT_LOG:
				break;
			case DT_DELTALOG:
				break;
			default:
				break;
			}
			if (d_tr[s*featuresCnt+f]<dmin[f]) {
				dmin[f]=d_tr[s*featuresCnt+f];
			}
			if (d_tr[s*featuresCnt+f]>dmax[f]) {
				dmax[f]=d_tr[s*featuresCnt+f];
			}
		}
	}


	//-- then, scale. ScaleMin/Max depend on the core, scaleM/P are specific for each feature
	for (f=0; f<featuresCnt; f++) {
		scaleM[f] = (scaleMax_-scaleMin_)/(dmax[f]-dmin[f]);
		scaleP[f] = scaleMax_-scaleM[f]*dmax[f];
	}

	for (s=0; s<steps; s++) {
		for (f=0; f<featuresCnt; f++) {
			d_trs[s*featuresCnt+f]=d_tr[s*featuresCnt+f]*scaleM[f]+scaleP[f];
			dbg->write(DBG_LEVEL_DET, "%d,%s,,,%f", 3, s, dtime[s], d_trs[s*featuresCnt+f]);
		}
		dbg->write(DBG_LEVEL_DET, "\n", 0);
	}

}
void sTimeSerie::unTrS(numtype scaleMin_, numtype scaleMax_) {
}

