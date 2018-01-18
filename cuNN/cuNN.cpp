#include "cuNN.h"

void D012_120(int d0, int d1, int d2, numtype* v) {
	numtype* newv=(numtype*)malloc(d0*d1*d2*sizeof(numtype));
	int i120, i012=0;
	for (int id0=0; id0<d0; id0++) {
		for (int id1=0; id1<d1; id1++) {
			for (int id2=0; id2<d2; id2++) {
				i120=id1*d2*d0+id2*d0+id0;
				newv[i120]=v[i012];
				i012++;
			}
		}
	}
	memcpy(v, newv, d0*d1*d2*sizeof(numtype));
	free(newv);
}
void D012_102(int d0, int d1, int d2, numtype* v) {
	numtype* newv=(numtype*)malloc(d0*d1*d2*sizeof(numtype));
	int i102, i012=0;
	for (int id0=0; id0<d0; id0++) {
		for (int id1=0; id1<d1; id1++) {
			for (int id2=0; id2<d2; id2++) {
				i102=id1*d0*d2+id0*d2+id2;
				newv[i102]=v[i012];
				i012++;
			}
		}
	}
	memcpy(v, newv, d0*d1*d2*sizeof(numtype));
	free(newv);
}
void SBF2BFS(int ds, int db, int df, numtype* v) {
	numtype* newv=(numtype*)malloc(ds*db*df*sizeof(numtype));
	int i=0;
	for(int s=0; s<ds; s++) {
		for(int b=0; b<db; b++) {
			for(int f=0; f<df; f++) {
				newv[b*ds*df+f*ds+s]=v[i];
				i++;
			}
		}
	}
	memcpy(v, newv, ds*db*df*sizeof(numtype));
	free(newv);
}

sNN::sNN(int sampleLen_, int predictionLen_, int featuresCnt_, int batchCnt_, int batchSamplesCnt_, char LevelRatioS_[60], bool useContext_, bool useBias_) {
	batchCnt=batchCnt_;
	batchSamplesCnt=batchSamplesCnt_;
	sampleLen=sampleLen_;
	predictionLen=predictionLen_;
	featuresCnt=featuresCnt_;
	useContext=useContext_;
	useBias=useBias_;

	InputCount=sampleLen_*featuresCnt_*batchSamplesCnt;
	OutputCount=predictionLen_*featuresCnt_*batchSamplesCnt;

	//-- 0. init CUDA/BLAS
	cublasH=new void*;
	cuRandH=new void*;
	if (myMemInit(cublasH, cuRandH)!=0) throw FAIL_INITCU;

	//-- 1. set Layout
	setLayout(LevelRatioS_);

	//-- 2. malloc neurons and weights on GPU
	if (myMalloc(&a, nodesCntTotal)!=0) throw FAIL_MALLOC_N;
	if (myMalloc(&F, nodesCntTotal)!=0) throw FAIL_MALLOC_N;
	if (myMalloc(&dF, nodesCntTotal)!=0) throw FAIL_MALLOC_N;
	if (myMalloc(&edF, nodesCntTotal)!=0) throw FAIL_MALLOC_N;
	if (myMalloc(&W, weightsCntTotal)!=0) throw FAIL_MALLOC_W;
	if (myMalloc(&dW, weightsCntTotal)!=0) throw FAIL_MALLOC_W;
	if (myMalloc(&dJdW, weightsCntTotal)!=0) throw FAIL_MALLOC_W;
	if (myMalloc(&e, nodesCnt[levelsCnt-1])!=0) throw FAIL_MALLOC_e;
	if (myMalloc(&u, nodesCnt[levelsCnt-1])!=0) throw FAIL_MALLOC_u;

	//-- device-based scalar value, to be used by reduction functions (sum, ssum, ...)
#ifdef USE_GPU
	if (myMalloc(&ss, 1)!=0) throw FAIL_MALLOC_SCALAR;
#endif

}
sNN::~sNN() {
	//!!!!!!!!!!!!!! create a myFree() functio to handle CUDA-based variables !
	free(weightsCnt);
	free(a);  free(F); free(dF); free(edF);
	free(W);
	//.....
	// free cublasH, cuRandH, curanddestroygenerator...
}

void sNN::setLayout(char LevelRatioS[60]) {
	int i, l, nl;
	int Levcnt;	// Levels count
	char** DescList=(char**)malloc(MAX_LEVELS*sizeof(char*)); for (i=0; i<MAX_LEVELS; i++) DescList[i]=(char*)malloc(256);

	//-- 0.1. levels and nodes count
	Levcnt = cslToArray(LevelRatioS, ',', DescList);

	for (i = 0; i<=Levcnt; i++) levelRatio[i] = (numtype)atof(DescList[i]);
	levelsCnt = (Levcnt+2);
	// set nodesCnt (single sample)
	nodesCnt[0] = InputCount;
	nodesCnt[levelsCnt-1] = OutputCount;
	for (nl = 0; nl<(levelsCnt-2); nl++) nodesCnt[nl+1] = (int)floor(nodesCnt[nl]*levelRatio[nl]);
	//-- add context neurons
	if (useContext) {
		for (nl = levelsCnt-1; nl>0; nl--) {
			nodesCnt[nl-1] += nodesCnt[nl];
		}
	}
	//-- add one bias neurons for each layer, except output layer
	if (useBias) {
		for (nl = 0; nl<(levelsCnt-1); nl++) nodesCnt[nl] += 1;
	}

	//-- 0.2. calc nodesCntTotal
	nodesCntTotal=0;
	for (l=0; l<levelsCnt; l++) nodesCntTotal+=nodesCnt[l];

	//-- 0.3. weights count
	weightsCntTotal=0;
	for (l=0; l<(levelsCnt-1); l++) {
		weightsCnt[l]=nodesCnt[l]/batchSamplesCnt*nodesCnt[l+1]/batchSamplesCnt;
		weightsCntTotal+=weightsCnt[l];
	}

	//-- 0.4. set first node and first weight for each layer
	for (l=0; l<levelsCnt; l++) {
		levelFirstNode[l]=0;
		levelFirstWeight[l]=0;
		for (int ll=0; ll<l; ll++) {
			levelFirstNode[l]+=nodesCnt[ll];
			levelFirstWeight[l]+=weightsCnt[ll];
		}
	}

	for (i=0; i<MAX_LEVELS; i++) free(DescList[i]);	free(DescList);

}

void sNN::setActivationFunction(int func_) {
	ActivationFunction=func_;
	switch (ActivationFunction) {
	case NN_ACTIVATION_TANH:
		scaleMin = -1;
		scaleMax = 1;
		break;
	case NN_ACTIVATION_EXP4:
		scaleMin = 0;
		scaleMax = 1;
		break;
	case NN_ACTIVATION_RELU:
		scaleMin = 0;
		scaleMax = 1;
		break;
	case NN_ACTIVATION_SOFTPLUS:
		scaleMin = 0;
		scaleMax = 1;
		break;
	default:
		scaleMin = -1;
		scaleMax = 1;
		break;
	}

}
int sNN::Activate(int level) {
	// sets dN
	int ret, retd;
	switch (ActivationFunction) {
	case NN_ACTIVATION_TANH:
		ret=Tanh(nodesCnt[level], &a[levelFirstNode[level]], &F[levelFirstNode[level]]);
		retd=dTanh(nodesCnt[level], &a[levelFirstNode[level]], &dF[levelFirstNode[level]]);
		break;
	case NN_ACTIVATION_EXP4:
		ret=Exp4(nodesCnt[level], &a[levelFirstNode[level]], &F[levelFirstNode[level]]);
		retd=dExp4(nodesCnt[level], &a[levelFirstNode[level]], &dF[levelFirstNode[level]]);
		break;
	case NN_ACTIVATION_RELU:
		ret=Relu(nodesCnt[level], &a[levelFirstNode[level]], &F[levelFirstNode[level]]);
		retd=dRelu(nodesCnt[level], &a[levelFirstNode[level]], &dF[levelFirstNode[level]]);
		break;
	case NN_ACTIVATION_SOFTPLUS:
		ret=SoftPlus(nodesCnt[level], &a[levelFirstNode[level]], &F[levelFirstNode[level]]);
		retd=dSoftPlus(nodesCnt[level], &a[levelFirstNode[level]], &dF[levelFirstNode[level]]);
		break;
	default:
		break;
	}
	return((ret==0&&retd==0)?0:-1);
}
int sNN:: calcErr() {
	//-- sets e, bte; adds squared sum(e) to tse
	if (Vdiff(nodesCnt[levelsCnt-1], &F[levelFirstNode[levelsCnt-1]], 1, u, 1, e)!=0) return -1;	// e=F[2]-u
	if (Vsum(nodesCnt[levelsCnt-1], e, &bte)!=0) return -1;											// bte=sum(e)
	if (Vssum(nodesCnt[levelsCnt-1], e, &se)!=0) return -1;											// se=ssum(e) 
	tse+=se;
	return 0;
}
int sNN::train(numtype* sample, numtype* target) {
	int l;
	char fname[MAX_PATH];
	DWORD batch_starttime, epoch_starttime;
	DWORD training_starttime=timeGetTime();

	int Ay, Ax, Astart, By, Bx, Bstart, Cy, Cx, Cstart;
	numtype* A; numtype* B; numtype* C;

	//-- Change the leading dimension in sample and target, from [Sample][Bar][Feature] to [Bar][Feature][Sample]
	int sampleCnt=batchCnt*batchSamplesCnt;	// 94

	SBF2BFS(sampleCnt, sampleLen,  featuresCnt, sample);
	SBF2BFS(sampleCnt, predictionLen, featuresCnt, target);

	//-- 0. Init
	
	//---- 0.1. Init Neurons (must set context neurons=0, at least for layer 0)
	if( Vinit(nodesCnt[0], &F[0], 0, 0) !=0) return -1;

	//---- 0.2. Init W
	for (l=0; l<(levelsCnt-1); l++) VinitRnd(weightsCnt[l], &W[levelFirstWeight[l]], -1/sqrtf((numtype)nodesCnt[l]), 1/sqrtf((numtype)nodesCnt[l]), cuRandH);
	//dumpData(weightsCntTotal, &W[0], "C:/temp/W.txt");

	//---- 0.3. Init dW
	if (Vinit(weightsCntTotal, dW, 0, 0)!=0) return -1;

	//-- 1. for every epoch, calc and display MSE
	for(int epoch=0; epoch<MaxEpochs; epoch++) {

		//-- timing
		epoch_starttime=timeGetTime();

		//-- 1.0. reset batch error and batch dW
		//Vinit(nodesCnt[levelsCnt-1], e, 0);
		//Vinit(weightsCntTotal, dW, 0);
		tse=0;

		//-- 1.1. train one batch at a time
		for (int b=0; b<batchCnt; b++) {

			//-- 1.1.1.  load samples + targets onto GPU
			if (loadBatchData(&F[0], &sample[b*InputCount], InputCount*sizeof(numtype) )!=0) return -1;
			if (loadBatchData(&u[0], &target[b*OutputCount], OutputCount*sizeof(numtype) )!=0) return -1;
			//dumpData(InputCount, &N[0], "C:/temp/F0.txt");
		
			//-- 1.1.2. Feed Forward ( W10[nc1 X nc0] X F0[nc0 X batchSize] => a1 [nc1 X batchSize] )
			for (l=0; l<(levelsCnt-1); l++) {
				int W10y=nodesCnt[l+1]/batchSamplesCnt;
				int W10x=nodesCnt[l]/batchSamplesCnt;
				int W10start= levelFirstWeight[l];
				int N0y=W10x;
				int N0x=batchSamplesCnt;
				int N0start= levelFirstNode[l];
				int N1start= levelFirstNode[l+1];
				
				//-- a[l+1]=F[l]*W[l]
				if (MbyM(cublasH, W10y, W10x, 1, false, &W[W10start], N0y, N0x, 1, false, &F[N0start], &a[N1start] ) !=0) return -1;
			
				//sprintf(fname, "C:/temp/F%d.txt", l); dumpData(nodesCnt[l], &N[levelFirstNode[l]], fname);
				//sprintf(fname, "C:/temp/F%d.txt", l+1); dumpData(nodesCnt[l+1], &N[levelFirstNode[l+1]], fname);

				//-- activation sets F[l+1] and dF[l+1]
				if(Activate(l+1)!=0) return -1;
			}
		
			//-- 1.1.3. Calc Error (sets e[], te, updates tse) for the whole batch
			if (calcErr()!=0) return -1;

			//sprintf(fname, "C:/temp/e.txt"); dumpData(nodesCnt[levelsCnt-1], &N[outNstart], fname);
			//sprintf(fname, "C:/temp/u.txt"); dumpData(nodesCnt[levelsCnt-1], &u[0], fname);

			//-- 1.1.4. BackPropagate, calc dJdW for the whole batch
			int sc=batchSamplesCnt;
			for (l = levelsCnt-1; l>0; l--) {
				if (l==(levelsCnt-1)) {
					//-- top level only
					if( VbyV2V(nodesCnt[l], e, &dF[levelFirstNode[l]], &edF[levelFirstNode[l]]) !=0) return -1;	// edF(l) = e * dF(l)
				} else {
					//-- lower levels
					Ay=nodesCnt[l+1]/sc;
					Ax=nodesCnt[l]/sc;
					Astart=levelFirstWeight[l];
					A=&W[Astart];
					By=nodesCnt[l+1]/sc;
					Bx=sc;
					Bstart=levelFirstNode[l+1];
					B=&edF[Bstart];
					Cy=Ax;	// because A gets transposed
					Cx=Bx;
					Cstart=levelFirstNode[l];
					C=&edF[Cstart];

					if (MbyM(cublasH, Ay, Ax, 1, true, A, By, Bx, 1, false, B, C)!=0) return -1;	// edF(l) = edF(l+1) * WT(l)
					if( VbyV2V(nodesCnt[l], &edF[levelFirstNode[l]], &dF[levelFirstNode[l]], &edF[levelFirstNode[l]]) !=0) return -1;	// edF(l) = edF(l) * dF(l)
				}
				
				//-- common	
				Ay=nodesCnt[l]/sc;
				Ax=sc;
				Astart=levelFirstNode[l];
				A=&edF[Astart];
				By=nodesCnt[l-1]/sc;
				Bx=sc;
				Bstart=levelFirstNode[l-1];
				B=&F[Bstart];
				Cy=Ay;
				Cx=By;// because B gets transposed
				Cstart=levelFirstWeight[l-1];
				C=&dJdW[Cstart];
				//Mfill(Ay*Ax, A, 0.1, 0.1);
				//Mfill(By*Bx, B, -0.1, -0.1);
				//Mprint(Ay, Ax, A, "A");
				//Mprint(By, Bx, B, "B");

				// dJdW(l-1) = edF(l) * F(l-1)
				if( MbyM(cublasH, Ay, Ax, 1, false, A, By, Bx, 1, true, B, C ) !=0) return -1;	
				


				//Mprint(Cy, Cx, C, "C");
				//sprintf(fname, "C:/temp/dJdW.txt"); dumpData(weightsCntTotal, dJdW, fname);
			}

			//-- 1.1.5. update weights for the whole batch
			//-- W = W - LR * dJdW
			//if (Vadd(weightsCntTotal, W, 1, dJdW, -LearningRate, W)!=0) return -1;

			//-- dW = LM*dW - LR*dJdW
			if (Vdiff(weightsCntTotal, dW, LearningMomentum, dJdW, LearningRate, dW) !=0) return -1;
			//sprintf(fname, "C:/temp/dW.txt"); dumpData(weightsCntTotal, dW, fname);

			//-- W = W + dW
			if (Vadd(weightsCntTotal, W, 1, dW, 1, W)!=0) return -1;

		}


		//-- 1.1. calc and display MSE
		//if (Vssum(nodesCnt[levelsCnt-1], e, &tse)!=0) return -1;
		mse=tse/batchCnt/nodesCnt[levelsCnt-1];

		printf("\repoch %d, MSE=%f, duration=%d ms", epoch, mse, (timeGetTime()-epoch_starttime));
	}
	printf("\nTraining complete. Elapsed time: %0.1f seconds.\n", (((float)timeGetTime()-(float)training_starttime)/(float)1000));
	//-
	return 0;
}