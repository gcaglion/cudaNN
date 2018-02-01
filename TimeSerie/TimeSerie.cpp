//#include <vld.h>
#include "TimeSerie.h"

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

int sTrainSet::buildFromTS(sTS* ts, int sampleLen_, int targetLen_, int intputFeature_[], char* outFileName) {

	int s, i, b, f;
	char LogFileName[MAX_PATH];
	FILE* LogFile=NULL;

	sampleLen=sampleLen_; targetLen=targetLen_;
	//-- THIS NEEDS TO CHANGE. We need to be able to create a training set with a subset of features for samples and a yet different one for targets
	featuresCnt=ts->featuresCnt;
	//--
	sampleSize=sampleLen*featuresCnt;
	targetSize=targetLen*featuresCnt;
	sampleCnt=ts->steps-sampleLen;
	sample=(numtype*)malloc(sampleCnt*sampleLen*featuresCnt*sizeof(numtype));
	target=(numtype*)malloc(sampleCnt*targetLen*featuresCnt*sizeof(numtype));
	sampleBFS=(numtype*)malloc(sampleCnt*sampleLen*featuresCnt*sizeof(numtype));
	targetBFS=(numtype*)malloc(sampleCnt*targetLen*featuresCnt*sizeof(numtype));

	if (ts->DebugParms->DebugLevel>0) {
		sprintf(LogFileName, ((outFileName==NULL)?"C:/temp/SlideArray.log":outFileName));
		LogFile = fopen(LogFileName, "w");
		fprintf(LogFile, "SampleId\t");
		for (int b=0; b<(sampleSize/featuresCnt); b++) {
			for (int f=0; f<featuresCnt; f++) {
				fprintf(LogFile, "  Bar%dF%d\t", b, f);
			}
		}
		fprintf(LogFile, "\t");
		for (int b=0; b<(targetSize/featuresCnt); b++) {
			for (int f=0; f<featuresCnt; f++) {
				fprintf(LogFile, "  Prd%dF%d\t", b, f);
			}
		}
		fprintf(LogFile, "\n");
		for (i=0; i<(1+sampleSize); i++) fprintf(LogFile, "---------\t");
		fprintf(LogFile, "\t");
		for (i=0; i<targetSize; i++) fprintf(LogFile, "---------\t");
		fprintf(LogFile, "\n");
	}

	int si, ti, sidx, tidx;
	si=0; ti=0;
	for (s=0; s<sampleCnt; s++) {
		//-- samples
		sidx=s*featuresCnt;
		if (ts->DebugParms->DebugLevel>0) fprintf(LogFile, "%d\t\t\t", s);
		//printf("\ns[%d] sidx=%d\n", s, sidx);
		for (b=0; b<sampleLen; b++) {
			for (f=0; f<featuresCnt; f++) {
				sample[si]=ts->d_trs[sidx];
				if (ts->DebugParms->DebugLevel>0) fprintf(LogFile, "%f\t", sample[si]);
				//printf("bar%df%d=%1.5f\n", b, f, ts->d[sidx]);
				sidx++;
				si++;
			}
		}
		if (ts->DebugParms->DebugLevel>0) fprintf(LogFile, "|\t");
		
		//-- targets
		tidx=sidx;
		//printf("\nt[%d] tidx=%d:\n", s, tidx);
		for (b=0; b<targetLen; b++) {
			for (f=0; f<featuresCnt; f++) {
				if (tidx==ts->len) {
					tidx-=featuresCnt;
				}
				target[ti]=ts->d_trs[tidx];
				if (ts->DebugParms->DebugLevel>0) fprintf(LogFile, "%f\t", target[ti]);
				//printf("bar%df%d=%1.5f\n", b, f, ts->d[tidx]);
				tidx++;
				ti++;
			}
		}
		if (ts->DebugParms->DebugLevel>0) fprintf(LogFile, "\n");
	}

	if (ts->DebugParms->DebugLevel>0) fclose(LogFile);

	return 0;
}
void sTrainSet::SBF2BFS(int batchCount_) {
	int i=0;
	for (int b=0; batchCount_; b++) {
		for (int bar=0; bar<sampleLen; bar++) {
			for (int f=0; f<featuresCnt; f++) {
				for (int s=0; s<sampleCnt; s++) {
					sampleBFS[i]=sample[b*sampleCnt*sampleLen*featuresCnt+s*sampleLen*featuresCnt+bar*featuresCnt+f];
					targetBFS[i]=target[b*sampleCnt*targetLen*featuresCnt+s*targetLen*featuresCnt+bar*featuresCnt+f];
					i++;
				}
			}
		}
	}
}
void sTrainSet::BFS2SBF(int batchCount_) {
	int i=0;
	for (int b=0; b<batchCount_; b++) {
		for (int s=0; s<sampleCnt; s++) {
			for (int bar=0; bar<sampleLen; bar++) {
				for (int f=0; f<featuresCnt; f++) {
					sample[i]=sampleBFS[b* sampleLen*featuresCnt*sampleCnt+bar*featuresCnt*sampleCnt+f*sampleCnt+s];
					i++;
				}
			}
		}
	}
}
