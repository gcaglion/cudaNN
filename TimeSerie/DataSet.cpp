#include "DataSet.h"

//-- sDataSet, constructors  /destructor
sDataSet::sDataSet(tTimeSerie* sourceTS_, int sampleLen_, int targetLen_, int batchSamplesCnt_, int selectedFeaturesCnt_, int* selectedFeature_, int* datafileBWFeature_, tDebugger* dbg_) {
	dbg=(dbg_==nullptr) ? (new tDebugger(DBG_LEVEL_ERR, DBG_DEST_FILE, new tFileInfo("DataSet.err"))) : dbg_;
	sourceTS=sourceTS_;
	//--
	selectedFeaturesCnt=selectedFeaturesCnt_;
	for (int f=0; f<selectedFeaturesCnt; f++) selectedFeature[f]=selectedFeature_[f];
	datafileBWFeature=(int*)malloc(2*sizeof(int));
	datafileBWFeature[0]=datafileBWFeature_[0]; datafileBWFeature[1]=datafileBWFeature_[1];
	//--
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
sDataSet::sDataSet(tParmsSource* parms, tTimeSerie* sourceTS_, tDebugger* dbg_) {
	dbg=(dbg_==nullptr) ? (new tDebugger(DBG_LEVEL_ERR, DBG_DEST_FILE, new tFileInfo("DataSets.err"))) : dbg_;
	sourceTS=sourceTS_;
/*	switch (sourceTS->set) {
	case TRAIN_SET:
		safeCallEB(parms->setKey(".Model.Data.TrainSet.DataSet"));
		break;
	case TEST_SET:
		safeCallEB(parms->setKey(".Model.Data.TestSet.DataSet"));
		break;
	case VALID_SET:
		safeCallEB(parms->setKey(".Model.Data.ValidationSet.DataSet"));
		break;
	default:
		break;
	}
*/
	parms->get(&batchSamplesCnt, "BatchSamplesCount");

	selectedFeature=(int*)malloc(MAX_DATA_FEATURES*sizeof(int));
	switch (sourceTS->sourceType) {
	case FILE_SOURCE:
		parms->get(&selectedFeature, "FileData.SelectedFeatures", &selectedFeaturesCnt);
		datafileBWFeature=(int*)malloc(2*sizeof(int));
		parms->get(&datafileBWFeature, "FileData.BWFeatures", new int);
		break;
	case FXDB_SOURCE:
		parms->get(&selectedFeature, "FXData.SelectedFeatures", &selectedFeaturesCnt);
		break;
	case MT4_SOURCE:
		//-- ...... ?? boh ??? ...
		break;
	default:
		break;
	}

}

sDataSet::~sDataSet() {
	free(selectedFeature);
	free(datafileBWFeature);
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
