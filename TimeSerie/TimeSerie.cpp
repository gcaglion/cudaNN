//#include <vld.h>
#include "TimeSerie.h"

//-- constructors / destructor
void sTS::sTScommon(int steps_, int featuresCnt_, tDbg* dbg_) {
	dbg=(dbg_==nullptr) ? (new tDbg(DBG_LEVEL_ERR, DBG_DEST_FILE, new tFileInfo("TimeSeries.err"))) : dbg_;
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
sTS::sTS(int steps_, int featuresCnt_, tDbg* dbg_) {
	sTScommon(steps_, featuresCnt_, dbg_);
}
sTS::sTS(tFXData* dataSource_, int steps_, char* date0_, int dt_, numtype scaleMin_, numtype scaleMax_, tDbg* dbg_){
	//-- 1. create
	sTScommon(steps_, FXDATA_FEATURESCNT, dbg_);	// no safeCall() because we don't set dbg, here
	//-- 2. load data
	safeCallEE(load(dataSource_, date0_));
	//-- 3. transform
	safeCallEE(transform(dt_));
	//-- 4. scale
	safeCallEE(scale(scaleMin_, scaleMax_));
}
sTS::sTS(tFileData* dataSource_, int steps_, char* date0_, int dt_, numtype scaleMin_, numtype scaleMax_, tDbg* dbg_){}
sTS::sTS(tMT4Data* dataSource_, int steps_, char* date0_, int dt_, numtype scaleMin_, numtype scaleMax_, tDbg* dbg_){}
sTS::~sTS() {
	free(d);
	free(bd);
	free(d_trs);
	free(d_tr);
	for (int i=0; i<len; i++) free(dtime[i]);
	free(dtime); free(bdtime);
	delete dbg;
}

bool sTS::LoadOHLCVdata(char* date0) {

	if (!OraConnect(dbg, FXData->db)) return false;
	if (!Ora_GetFlatOHLCV(dbg, FXData->db, FXData->Symbol, FXData->TimeFrame, date0, this->steps, this->dtime, this->d, this->bdtime, this->bd)) return false;

	return true;
}
void sTS::load(tFXData* tsFXData_, char* pDate0) {
	FXData=tsFXData_;
	sourceType=SOURCE_DATA_FROM_FXDB;
	if (!LoadOHLCVdata(pDate0)) throwE("pDate0=%s", 1, pDate0);
}
void sTS::load(tFileData* tsFileData, char* pDate0) {
	throwE("", 0);
}
void sTS::load(tMT4Data* tsMT4Data, char* pDate0) {
	throwE("", 0);
}
void sTS::dump(char* dumpFileName) {
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
void sTS::transform(int dt_) {
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
void sTS::scale(numtype scaleMin_, numtype scaleMax_) {
	//-- ScaleMin/Max depend on the core, scaleM/P are specific for each feature

	if (!hasTR) throwE("-- must transform before scaling! ---", 0);

	for (int f=0; f<featuresCnt; f++) {
		scaleM[f] = (scaleMax_-scaleMin_)/(dmax[f]-dmin[f]);
		scaleP[f] = scaleMax_-scaleM[f]*dmax[f];
	}

	for (int s=0; s<steps; s++) {
		for (int f=0; f<featuresCnt; f++) {
			d_trs[s*featuresCnt+f]=d_tr[s*featuresCnt+f]*scaleM[f]+scaleP[f];
		}
	}

	hasTRS=true;
}

void sTS::TrS(int dt_, numtype scaleMin_, numtype scaleMax_) {
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
void sTS::unTrS(numtype scaleMin_, numtype scaleMax_) {
}

int getMcol_cpu(int Ay, int Ax, numtype* A, int col, numtype* oCol) {
	for (int y=0; y<Ay; y++) oCol[y]=A[y*Ax+col];
	return 0;
}

sDataSet::sDataSet(sTS* sourceTS_, int sampleLen_, int targetLen_, int selectedFeaturesCnt_, int* selectedFeature_, int batchSamplesCnt_, tDbg* dbg_) {
	dbg=(dbg_==nullptr) ? (new tDbg(DBG_LEVEL_ERR, DBG_DEST_FILE, new tFileInfo("DataSet.err"))) : dbg_;
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
void sDataSet::buildFromTS(tTS* ts) {

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
