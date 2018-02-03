//#include <vld.h>
#include "TimeSerie.h"

void SBF2BFScommon(int db, int ds, int dbar, int df, numtype* iv, numtype* ov) {
	int i=0;
	for (int b=0; b<db; b++) {
		for (int bar=0; bar<dbar; bar++) {
			for (int f=0; f<df; f++) {
				for (int s=0; s<ds; s++) {
					ov[i]=iv[b*ds*dbar*df+s*dbar*df+bar*df+f];
					i++;
				}
			}
		}
	}
}

int sTS::LoadOHLCVdata(char* date0) {

	if (OraConnect(DebugParms, FXData->FXDB)!=0) return -1;
	if (Ora_GetFlatOHLCV(DebugParms, FXData->FXDB->DBCtx, FXData->Symbol, FXData->TimeFrame, date0, this->steps, this->dtime, this->d, this->bdtime, this->bd)!=0) return -1;

	return 0;
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

int sTS::TrS(int dt_, numtype scaleMin_, numtype scaleMax_) {
	dt=dt_;
	FILE* ftrs=nullptr;
	if (DebugParms->DebugLevel>1) {
		ftrs=fopen("c:/temp/ftrs.csv", "w");
		fprintf(ftrs, "i,datetime,O,Otr,Otrs,H,Htr,Htrs,L,Ltr,Ltrs,C,Ctr,Ctrs,V,Vtrs \n");
		fprintf(ftrs, "%d,%s,%f,,,%f,,,%f,,,%f,,,%f \n", -1, bdtime, bd[0], bd[1], bd[2], bd[3], bd[4]);
	}

	int s, f;
	//-- first, transform
	for (s=0; s<steps; s++) {
		if (DebugParms->DebugLevel>1) fprintf(ftrs, "%d, %s", s, dtime[s]);
		for (f=0; f<featuresCnt; f++) {
			if (DebugParms->DebugLevel>1) fprintf(ftrs, ",%f", d[s*featuresCnt+f]);
			switch (dt) {
			case DT_NONE:
				break;
			case DT_DELTA:
				if (s>0) {
					d_trs[s*featuresCnt+f]=d[s*featuresCnt+f]-d[(s-1)*featuresCnt+f];
				} else {
					d_trs[s*featuresCnt+f]=d[s*featuresCnt+f]-bd[f];
				}
				if (DebugParms->DebugLevel>1) fprintf(ftrs, ",%f,", d_trs[s*featuresCnt+f]);
				break;
			case DT_LOG:
				break;
			case DT_DELTALOG:
				break;
			default:
				break;
			}
			if (d_trs[s*featuresCnt+f]<dmin[f]) {
				dmin[f]=d_trs[s*featuresCnt+f];
			}
			if (d_trs[s*featuresCnt+f]>dmax[f]) {
				dmax[f]=d_trs[s*featuresCnt+f];
			}
		}
		if (DebugParms->DebugLevel>1) fprintf(ftrs, "\n");
	}
	if (DebugParms->DebugLevel>1) fprintf(ftrs, "\n");

	if (DebugParms->DebugLevel>1) fprintf(ftrs, "min:,,%f,,,%f,,,%f,,,%f,,,%f \n", dmin[0], dmin[1], dmin[2], dmin[3], dmin[4]);
	if (DebugParms->DebugLevel>1) fprintf(ftrs, "max:,,%f,,,%f,,,%f,,,%f,,,%f \n", dmax[0], dmax[1], dmax[2], dmax[3], dmax[4]);
	if (DebugParms->DebugLevel>1) fprintf(ftrs, "\n");

	//-- then, scale. ScaleMin/Max depend on the core, scaleM/P are specific for each feature
	for (f=0; f<featuresCnt; f++) {
		scaleM[f] = (scaleMax_-scaleMin_)/(dmax[f]-dmin[f]);
		scaleP[f] = scaleMax_-scaleM[f]*dmax[f];
	}
	if (DebugParms->DebugLevel>1) fprintf(ftrs, "scaleM:,,%f,,,%f,,,%f,,,%f,,,%f \n", scaleM[0], scaleM[1], scaleM[2], scaleM[3], scaleM[4]);
	if (DebugParms->DebugLevel>1) fprintf(ftrs, "scaleP:,,%f,,,%f,,,%f,,,%f,,,%f \n", scaleP[0], scaleP[1], scaleP[2], scaleP[3], scaleP[4]);

	for (s=0; s<steps; s++) {
		for (f=0; f<featuresCnt; f++) {
			d_trs[s*featuresCnt+f]=d_trs[s*featuresCnt+f]*scaleM[f]+scaleP[f];
			if (DebugParms->DebugLevel>1) fprintf(ftrs, "%d,%s,,,%f", s, dtime[s], d_trs[s*featuresCnt+f]);
		}
		if (DebugParms->DebugLevel>1) fprintf(ftrs, "\n");
	}
	if (DebugParms->DebugLevel>1) fclose(ftrs);

	return 0;
}

int sTS::unTrS(numtype scaleMin_, numtype scaleMax_) {
	return 0;
}

void sDataSet::dump(char* filename) {
	int s, i, b, f;
	char LogFileName[MAX_PATH];
	FILE* LogFile=NULL;
	sprintf(LogFileName, ((filename==nullptr) ? "C:/temp/DataSet.log" : filename));

	LogFile = fopen(LogFileName, "w");
	fprintf(LogFile, "SampleId\t");
	for ( b=0; b<(sampleLen); b++) {
		for ( f=0; f<selectedFeaturesCnt; f++) {
			fprintf(LogFile, "  Bar%dF%d\t", b, selectedFeature[f]);
		}
	}
	fprintf(LogFile, "\t");
	for ( b=0; b<(targetLen); b++) {
		for ( f=0; f<selectedFeaturesCnt; f++) {
			fprintf(LogFile, "  Prd%dF%d\t", b, selectedFeature[f]);
		}
	}
	fprintf(LogFile, "\n");
	for (i=0; i<(1+sampleSize); i++) fprintf(LogFile, "---------\t");
	fprintf(LogFile, "\t");
	for (i=0; i<targetSize; i++) fprintf(LogFile, "---------\t");
	fprintf(LogFile, "\n");

	int si, ti, sidx, tidx;
	si=0; ti=0;
	for (s=0; s<sampleCnt; s++) {
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
int sDataSet::buildFromTS(sTS* ts) {

	int s, i, b, f;

	int si, ti, sidx, tidx;
	si=0; ti=0;
	for (s=0; s<sampleCnt; s++) {
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
					if (tidx==ts->len) {
						tidx-=ts->featuresCnt;
					}
					target[ti]=ts->d_trs[tidx];
					ti++;
				}
				tidx++;
			}
		}
	}

	return 0;
}

void sDataSet::SBF2BFS() {

	int barCnt;
	//-- first, sample
	barCnt=sampleLen;
	SBF2BFScommon(batchCnt, batchSamplesCnt, barCnt, selectedFeaturesCnt, sample, sampleBFS);
	//-- then, target
	barCnt=targetLen;
	SBF2BFScommon(batchCnt, batchSamplesCnt, barCnt, selectedFeaturesCnt, target, targetBFS);

}
void sDataSet::BFS2SBF() {
	int i=0;
	for (int b=0; b<batchCnt; b++) {
		for (int s=0; s<batchSamplesCnt; s++) {
			for (int bar=0; bar<sampleLen; bar++) {
				for (int f=0; f<sourceTS->featuresCnt; f++) {
					sample[i]=sampleBFS[b* sampleLen*sourceTS->featuresCnt*batchSamplesCnt+bar*sourceTS->featuresCnt*batchSamplesCnt+f*batchSamplesCnt+s];
					i++;
				}
			}
		}
	}
}

