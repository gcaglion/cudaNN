//#include <vld.h>
#include "TimeSerie.h"

//-- sTimeSerie, constructors / destructor
void sTimeSerie::sTimeSeriecommon(int steps_, int featuresCnt_, tDebugger* dbg_) {
	dbg=(dbg_==nullptr) ? (new tDebugger(DBG_LEVEL_ERR, DBG_DEST_FILE, new tFileInfo("TimeSeries.err"))) : dbg_;
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

	//-- enums

}
sTimeSerie::sTimeSerie(int steps_, int featuresCnt_, tDebugger* dbg_) {
	sTimeSeriecommon(steps_, featuresCnt_, dbg_);
}
sTimeSerie::sTimeSerie(tFXData* dataSource_, int steps_, char* date0_, int dt_, tDebugger* dbg_){
	//-- 1. create
	sTimeSeriecommon(steps_, FXDATA_FEATURESCNT, dbg_);	// no safeCall() because we don't set dbg, here
	//-- 2. load data
	safeCallEE(load(dataSource_, date0_));
	//-- 3. transform
	safeCallEE(transform(dt_));
}
sTimeSerie::sTimeSerie(tFileData* dataSource_, int steps_, int featuresCnt_, char* date0_, int dt_, tDebugger* dbg_){
	featuresCnt=featuresCnt_;
}
sTimeSerie::sTimeSerie(tMT4Data* dataSource_, int steps_, char* date0_, int dt_, tDebugger* dbg_){
}
sTimeSerie::sTimeSerie(tParmsSource* parms, int set_, tDebugger* dbg_){
	dbg=(dbg_==nullptr) ? (new tDebugger(DBG_LEVEL_ERR, DBG_DEST_FILE, new tFileInfo("TimeSeries.err"))) : dbg_;
	set=set_;
	
	//-- 1. set xml section according to set (Train/Test/Validation)
	switch (set) {
	case TRAIN_SET:
		parms->gotoKey("Data.Train.TimeSerie");
		break;
	case TEST_SET:
		parms->gotoKey("Data.Test.TimeSerie");
		break;
	case VALID_SET:
		parms->gotoKey("Data.Validation.TimeSerie");
		break;
	default:
		break;
	}

	//-- 1.1. First, create sub-object DataSource from sub-key <DataSource>
	parms->get(&sourceType, "DataSourceType", true);
	if (sourceType==FXDB_SOURCE) {
		safeCallEE(fxData=new tFXData(parms));
	} else if (sourceType==FILE_SOURCE) {
		safeCallEE(fileData=new tFileData(parms));
	} else if (sourceType==MT4_SOURCE) {
		safeCallEE(mt4Data=new tMT4Data(parms));
	} else {
		throwE("invalid DataSourceType in section %s", 1, "SALCAZZO");
	}
	//-- 1.2 <TimeSerie> root parms

}
sTimeSerie::~sTimeSerie() {
	free(d);
	free(bd);
	free(d_trs);
	free(d_tr);
	for (int i=0; i<len; i++) free(dtime[i]);
	free(dtime); free(bdtime);
	delete dbg;
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
void sTimeSerie::load(tMT4Data* tsMT4Data, char* pDate0) {
	throwE("", 0);
}
void sTimeSerie::dump(char* dumpFileName) {
	int s, f;
	tFileInfo* fdump; safeCallEE(fdump=new tFileInfo(dumpFileName));
	fprintf(fdump->handle, "i, datetime");
	for (f=0; f<featuresCnt; f++) fprintf(fdump->handle, ",F%d_orig,F%d_tr,F%d_trs", f, f, f);
	fprintf(fdump->handle, "\n%d,%s", -1, bdtime);
	for (f=0; f<featuresCnt; f++) {
		fprintf(fdump->handle, ",%f", bd[f]);
		for (int ff=0; ff<(featuresCnt-3); ff++) fprintf(fdump->handle, ",");
	}

	for (s=0; s<steps; s++) {
		fprintf(fdump->handle, "\n%d, %s", s, dtime[s]);
		for (f=0; f<featuresCnt; f++) {
			fprintf(fdump->handle, ",%f", d[s*featuresCnt+f]);
			if (hasTR) {
				fprintf(fdump->handle, ",%f", d_tr[s*featuresCnt+f]);
			} else {
				fprintf(fdump->handle, ",");
			}
			if (hasTRS) {
				fprintf(fdump->handle, ",%f", d_trs[s*featuresCnt+f]);
			} else {
				fprintf(fdump->handle, ",");
			}
		}
	}
	fprintf(fdump->handle, "\n");

	if (hasTR) {
		fprintf(fdump->handle, "\ntr-min:");
		for (f=0; f<featuresCnt; f++) fprintf(fdump->handle, ",,,%f", dmin[f]);
		fprintf(fdump->handle, "\ntr-max:");
		for (f=0; f<featuresCnt; f++) fprintf(fdump->handle, ",,,%f", dmax[f]);
		fprintf(fdump->handle, "\n");
	}
	if (hasTRS) {
		fprintf(fdump->handle, "\nscaleM:");
		for (f=0; f<featuresCnt; f++) fprintf(fdump->handle, ",,,%f", scaleM[f]);
		fprintf(fdump->handle, "\nscaleP:");
		for (f=0; f<featuresCnt; f++) fprintf(fdump->handle, ",,,%f", scaleP[f]);
		fprintf(fdump->handle, "\n");

		//fprintf(fdump->handle, "scaleM:,,%f,,,%f,,,%f,,,%f,,,%f \n", scaleM[0], scaleM[1], scaleM[2], scaleM[3], scaleM[4]);
		//fprintf(fdump->handle, "scaleP:,,%f,,,%f,,,%f,,,%f,,,%f \n", scaleP[0], scaleP[1], scaleP[2], scaleP[3], scaleP[4]);
	}

	delete fdump;

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
	tFileInfo* ftrs=nullptr; safeCallEE(ftrs=new tFileInfo("TransformScale.out"));

	int s, f;
	//-- first, transform
	for (s=0; s<steps; s++) {
		for (f=0; f<featuresCnt; f++) {
			if (dbg->level>1) fprintf(ftrs->handle, ",%f", d[s*featuresCnt+f]);
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
			if (dbg->level>1) fprintf(ftrs->handle, "%d,%s,,,%f", s, dtime[s], d_trs[s*featuresCnt+f]);
		}
		if (dbg->level>1) fprintf(ftrs->handle, "\n");
	}
	delete ftrs;

}
void sTimeSerie::unTrS(numtype scaleMin_, numtype scaleMax_) {
}

//-- UberSet
/*
sUberSetParms::sUberSetParms(tParmsSource* parms_, int set, tDebugger* dbg_) {
	parms=parms_; dbg=dbg_;

	SelectedFeature=(int*)malloc(MAX_DATA_FEATURES*sizeof(int));

	//-- 0. Data model parms (set-invariant)
	if(set==US_MODEL){
		parms->sectionSet("Data.Model");
		parms->getx(&SampleLen, "SampleLen");
		parms->getx(&PredictionLen, "PredictionLen");
		parms->getx(&FeaturesCnt, "FeaturesCount");
		return;
	}

	//-- Cores parameters
	if (set==US_CORE_NN) {
		parms->sectionSet("Engine.Core");
		parms->getx(&NN->useContext, "Topology.UseContext");
		parms->getx(&NN->useBias, "Topology.UseBias");
		parms->getx(&NN->levelRatio, "Topology.LevelRatio", false, &NN->levelsCnt); NN->levelsCnt+=2;
		int ac;
		parms->getx(&NN->ActivationFunction, "Topology.LevelActivation", true, &ac);
		if (ac!=(NN->levelsCnt)) throwE("Levels count (%d) and LevelActivation count (%d) are incompatible.", NN->levelsCnt, ac);
		parms->getx(&NN->MaxEpochs, "Training.MaxEpochs");
		parms->getx(&NN->TargetMSE, "Training.TargetMSE");
		parms->getx(&NN->NetSaveFreq, "Training.NetSaveFrequency");
		parms->getx(&NN->StopOnDivergence, "Training.StopOnDivergence");
		parms->getx(&NN->BP_Algo, "Training.BP_Algo");
		parms->getx(&NN->LearningRate, "Training.BP_Std.LearningRate");
		parms->getx(&NN->LearningMomentum, "Training.BP_Std.LearningMomentum");
		return;
	}
	if (set==US_CORE_SVM) {
		return;
	}
	if (set==US_CORE_GA) {
		return;
	}

	//==========	TimeSeries-DataSets type sets
	
	//-- set-specific parms
	if (set==US_TRAIN) {
		parms->getx(&doTrainRun, "Data.Train", "doCheck");
	}
	if (set==US_TEST) {
		//-- load saved Net?
		parms->sectionSet("Data.Test");
		parms->getx(&runFromSavedNet, "RunFromSavedNet");
		if (runFromSavedNet) {
			parms->getx(&testWpid, "RunFromSavedNet.ProcessId");
			parms->getx(&testWtid, "RunFromSavedNet.ThreadId");
		}
	}
	if (set==US_VALID) {
		parms->sectionSet("Data.Validation");
	}
	
	//-- common to all TimeSeries-DataSets type sets
	if(set==US_TRAIN ||set==US_TEST || set==US_VALID){
		parms->getx(&doIt, "doIt");
		if (doIt) {
			//-- 1.1. TimeSerie, common
			parms->getx(TSdate0, "TimeSerie.Date0");
			parms->getx(&TShistoryLen, "TimeSerie.HistoryLen");
			parms->getx(&TS_DT, "TimeSerie.DataTransformation", enumlist);
			parms->getx(&TS_BWcalc, "TimeSerie.BWcalc");
			parms->getx(&TS_DS_type, "TimeSerie.DataSource.DataSourceType", enumlist);
			//-- 1.2. TimeSerie, datasource-specific
			if (TS_DS_type==SOURCE_DATA_FROM_FXDB) {
				parms->getx(TS_DS_FX_DBUser, "TimeSerie.DataSource.FXData.DBUser");
				parms->getx(TS_DS_FX_DBPassword, "TimeSerie.DataSource.FXData.DBPassword");
				parms->getx(TS_DS_FX_DBConnString, "TimeSerie.DataSource.FXData.DBConnString");
				parms->getx(TS_DS_FX_Symbol, "TimeSerie.DataSource.FXData.Symbol");
				parms->getx(TS_DS_FX_TimeFrame, "TimeSerie.DataSource.FXData.TimeFrame");
				parms->getx(&TS_DS_FX_IsFilled, "TimeSerie.DataSource.FXData.IsFilled");
				safeCallEE(TS_DS_FX_DB=new tDBConnection(TS_DS_FX_DBUser, TS_DS_FX_DBPassword, TS_DS_FX_DBConnString));
				safeCallEE(TS_DS_FX=new tFXData(TS_DS_FX_DB, TS_DS_FX_Symbol, TS_DS_FX_TimeFrame, TS_DS_FX_IsFilled));
				safeCallEE(TS=new tTimeSerie(TS_DS_FX, TShistoryLen, TSdate0, TS_DT));
			} else if (TS_DS_type==SOURCE_DATA_FROM_FILE) {
				parms->getx(TS_DS_File_FullName, "TimeSerie.DataSource.FileData.FileFullName");
				parms->getx(&TS_DS_File_FieldSep, "TimeSerie.DataSource.FileData.FieldSep", enumlist);
				parms->getx(&TS_DS_File_BWcol, "TimeSerie.DataSource.FileData.BWFeatureColumns");
				safeCallEE(TS_DS_File=new tFileData(new tFileInfo(TS_DS_File_FullName, FILE_MODE_READ), TS_DS_File_FieldSep, TS_BWcalc, TS_DS_File_BWcol[HIGH], TS_DS_File_BWcol[LOW]));
				safeCallEE(TS=new tTimeSerie(TS_DS_File, TS_DS_File->featuresCnt, TShistoryLen, TSdate0, TS_DT));
			}
			//-- 1.2. DataSet
			if (TS_DS_type==SOURCE_DATA_FROM_FXDB) {
				parms->getx(&SelectedFeature, "DataSet.FXData.SelectedFeatures", enumlist, &SelectedFeaturesCnt);
			} else if (TS_DS_type==SOURCE_DATA_FROM_FILE) {
				parms->getx(&SelectedFeature, "DataSet.FileData.SelectedFeatures", false, &SelectedFeaturesCnt);
			}
			parms->getx(&BatchSamplesCnt, "DataSet.BatchSamplesCount");
			safeCallEE(DataSet=new tDataSet(TS, SampleLen, PredictionLen, SelectedFeaturesCnt, SelectedFeature, BatchSamplesCnt));
		}
	}
}
sUberSetParms::~sUberSetParms() {
	free(SelectedFeature);
}
*/

//-- sDataSet, constructors  /destructor
sDataSet::sDataSet(sTimeSerie* sourceTS_, int sampleLen_, int targetLen_, int selectedFeaturesCnt_, int* selectedFeature_, int batchSamplesCnt_, tDebugger* dbg_) {
	dbg=(dbg_==nullptr) ? (new tDebugger(DBG_LEVEL_ERR, DBG_DEST_FILE, new tFileInfo("DataSet.err"))) : dbg_;
	sourceTS=sourceTS_;
	selectedFeaturesCnt=selectedFeaturesCnt_; selectedFeature=selectedFeature_;
	sampleLen=sampleLen_;
	targetLen=targetLen_;
	samplesCnt=sourceTS->steps-sampleLen-targetLen;// +1;
	if (samplesCnt<1) throwE("Not Enough Data. samplesCnt=%d", 1, samplesCnt);
	batchSamplesCnt=batchSamplesCnt_;
	batchCnt=samplesCnt/batchSamplesCnt;
	if ((batchCnt*batchSamplesCnt)!=samplesCnt) throwE("Wrong Batch Size. samplesCnt=%d, batchSamplesCnt=%d", 2, samplesCnt, batchSamplesCnt);

	sample=(numtype*)malloc(samplesCnt*sampleLen*selectedFeaturesCnt*sizeof(numtype));
	target=(numtype*)malloc(samplesCnt*targetLen*selectedFeaturesCnt*sizeof(numtype));
	prediction=(numtype*)malloc(samplesCnt*targetLen*selectedFeaturesCnt*sizeof(numtype));
	sampleBFS=(numtype*)malloc(samplesCnt*sampleLen*selectedFeaturesCnt*sizeof(numtype));
	targetBFS=(numtype*)malloc(samplesCnt*targetLen*selectedFeaturesCnt*sizeof(numtype));
	predictionBFS=(numtype*)malloc(samplesCnt*targetLen*selectedFeaturesCnt*sizeof(numtype));
	//--
	targetSFB=(numtype*)malloc(samplesCnt*targetLen*selectedFeaturesCnt*sizeof(numtype));
	predictionSFB=(numtype*)malloc(samplesCnt*targetLen*selectedFeaturesCnt*sizeof(numtype));
	//--
	target0=(numtype*)malloc(samplesCnt*selectedFeaturesCnt*sizeof(numtype));
	prediction0=(numtype*)malloc(samplesCnt*selectedFeaturesCnt*sizeof(numtype));

	//-- fill sample/target data right at creation time. TS has data in SBF format
	buildFromTS(sourceTS);

	for (int b=0; b<batchCnt; b++) {
		//-- populate BFS sample/target for every batch
		SBF2BFS(b, sampleLen, sample, sampleBFS);
		SBF2BFS(b, targetLen, target, targetBFS);
		//-- populate SFB targets, too
		BFS2SFB(b, targetLen, targetBFS, targetSFB);
	}
}
sDataSet::sDataSet(tParmsSource* parms, sTimeSerie* sourceTS_, tDebugger* dbg_) {
	dbg=(dbg_==nullptr) ? (new tDebugger(DBG_LEVEL_ERR, DBG_DEST_FILE, new tFileInfo("DataSets.err"))) : dbg_;
	sourceTS=sourceTS_;

	switch (sourceTS->set) {
	case TRAIN_SET:
		parms->gotoKey("Data.Train.DataSet");
		break;
	case TEST_SET:
		parms->gotoKey("Data.Test.DataSet");
		break;
	case VALID_SET:
		parms->gotoKey("Data.Validation.DataSet");
		break;
	default:
		break;
	}
	parms->get(&batchSamplesCnt, "BatchSamplesCount");
	switch (sourceTS->sourceType) {
	case FILE_SOURCE:
		parms->get(&selectedFeature, "SelectedFeatures", false, false, &selectedFeaturesCnt);
		break;
	case FXDB_SOURCE:
		parms->get(&selectedFeature, "SelectedFeatures", false, true, &selectedFeaturesCnt);
		break;
	case MT4_SOURCE:
		//-- ...... ?? boh ??? ...
		break;
	default:
		break;
	}

}

sDataSet::~sDataSet() {
	free(sample);
	if (target!=nullptr) free(target);
	free(prediction);
	free(sampleBFS);
	free(targetBFS);
	free(predictionBFS);
	free(target0);
	free(prediction0);

	delete dbg;
}
//-- sDataSet, other methods
void sDataSet::dump(char* filename) {
	int s, i, b, f;
	char LogFileName[MAX_PATH];
	FILE* LogFile=NULL;
	sprintf(LogFileName, ((filename==nullptr) ? "C:/temp/DataSet.log" : filename));

	LogFile = fopen(LogFileName, "w");
	fprintf(LogFile, "SampleId\t");
	for (b=0; b<(sampleLen); b++) {
		for (f=0; f<selectedFeaturesCnt; f++) {
			fprintf(LogFile, "  Bar%dF%d\t", b, selectedFeature[f]);
		}
	}
	fprintf(LogFile, "\t");
	for (b=0; b<(targetLen); b++) {
		for (f=0; f<selectedFeaturesCnt; f++) {
			fprintf(LogFile, "  Prd%dF%d\t", b, selectedFeature[f]);
		}
	}
	fprintf(LogFile, "\n");
	for (i=0; i<(1+(sampleLen*selectedFeaturesCnt)); i++) fprintf(LogFile, "---------\t");
	fprintf(LogFile, "\t");
	for (i=0; i<(targetLen*selectedFeaturesCnt); i++) fprintf(LogFile, "---------\t");
	fprintf(LogFile, "\n");

	int si, ti, sidx, tidx;
	si=0; ti=0;
	for (s=0; s<samplesCnt; s++) {
		//-- samples
		sidx=s*sourceTS->featuresCnt;
		fprintf(LogFile, "%d\t\t\t", s);
		for (b=0; b<sampleLen; b++) {
			for (f=0; f<sourceTS->featuresCnt; f++) {
				if (isSelected(f)) {
					fprintf(LogFile, "%f\t", sample[si]);
					si++;
				}
				sidx++;
			}
		}
		fprintf(LogFile, "|\t");

		//-- targets
		tidx=sidx;
		for (b=0; b<targetLen; b++) {
			for (f=0; f<sourceTS->featuresCnt; f++) {
				if (isSelected(f)) {
					if (tidx==sourceTS->len) {
						tidx-=sourceTS->featuresCnt;
					}
					fprintf(LogFile, "%f\t", target[ti]);
					ti++;
				}
				tidx++;
			}
		}
		fprintf(LogFile, "\n");
	}
	fclose(LogFile);
}
bool sDataSet::isSelected(int ts_f) {
	for (int ds_f=0; ds_f<selectedFeaturesCnt; ds_f++) {
		if (selectedFeature[ds_f]==ts_f) return true;
	}
	return false;
}
void sDataSet::buildFromTS(tTimeSerie* ts) {

	int s, b, f;

	int si, ti, sidx, tidx;
	si=0; ti=0;
	for (s=0; s<samplesCnt; s++) {
		//-- samples
		sidx=s*ts->featuresCnt;
		for (b=0; b<sampleLen; b++) {
			for (f=0; f<ts->featuresCnt; f++) {
				if (isSelected(f)) {
					sample[si]=ts->d_trs[sidx];
					si++;
				}
				sidx++;
			}
		}
		//-- targets
		tidx=sidx;
		for (b=0; b<targetLen; b++) {
			for (f=0; f<ts->featuresCnt; f++) {
				if (isSelected(f)) {
					target[ti]=ts->d_trs[tidx];
					ti++;
				}
				tidx++;
			}
		}
	}

}
void sDataSet::SBF2BFS(int batchId, int barCnt, numtype* fromSBF, numtype* toBFS) {
	int S=batchSamplesCnt;
	int F=selectedFeaturesCnt;
	int B=barCnt;
	int idx;
	int idx0=batchId*B*F*S;
	int i=idx0;
	for (int bar=0; bar<B; bar++) {												// i1=bar	l1=B
		for (int f=0; f<F; f++) {										// i2=f		l2=F
			for (int s=0; s<S; s++) {										// i3=s		l3=S
				idx=idx0+s*F*B+bar*F+f;
				toBFS[i]=fromSBF[idx];
				i++;
			}
		}
	}
}
void sDataSet::BFS2SBF(int batchId, int barCnt, numtype* fromBFS, numtype* toSBF) {
	int S=batchSamplesCnt;
	int F=selectedFeaturesCnt;
	int B=barCnt;
	int idx;
	int idx0=batchId*B*F*S;
	int i=idx0;
	for (int s=0; s<S; s++) {												// i1=s		l1=S
		for (int bar=0; bar<B; bar++) {											// i2=bar	l1=B
			for (int f=0; f<F; f++) {									// i3=f		l3=F
				idx=idx0+bar*F*S+f*S+s;
				toSBF[i]=fromBFS[idx];
				i++;
			}
		}
	}

}
void sDataSet::BFS2SFBfull(int barCnt, numtype* fromBFS, numtype* toSFB) {
	int S=batchSamplesCnt;
	int F=selectedFeaturesCnt;
	int B=barCnt;
	int i, idx, idx0;
	for (int batchId=0; batchId<batchCnt; batchId++) {
		idx0=batchId*B*F*S;
		i=idx0;
		for (int s=0; s<S; s++) {												// i1=s		l1=S
			for (int f=0; f<F; f++) {										// i2=f		l2=F
				for (int bar=0; bar<B; bar++) {										// i3=bar	l3=B
					idx=idx0+bar*F*S+f*S+s;
					toSFB[i]=fromBFS[idx];
					i++;
				}
			}
		}
	}

}
void sDataSet::BFS2SFB(int batchId, int barCnt, numtype* fromBFS, numtype* toSFB) {
	int S=batchSamplesCnt;
	int F=selectedFeaturesCnt;
	int B=barCnt;
	int idx;
	int idx0=batchId*B*F*S;
	int i=idx0;
	for (int s=0; s<S; s++) {												// i1=s		l1=S
		for (int f=0; f<F; f++) {										// i2=f		l2=F
			for (int bar=0; bar<B; bar++) {										// i3=bar	l3=B
				idx=idx0+bar*F*S+f*S+s;
				toSFB[i]=fromBFS[idx];
				i++;
			}
		}
	}

}
