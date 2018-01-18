#pragma once

#include "../CommonEnv.h"
#include "../MyUtils/MyUtils.h"
#include <stdio.h>
#include <time.h>
#include <math.h>

typedef struct s_matrix {
	int my;
	int mx;
	numtype* m;

	s_matrix(int my_, int mx_, bool init_=false, numtype val0=0, numtype inc=0 ) {
		my=my_; mx=mx_;
		m=(numtype*)malloc(my*mx*sizeof(numtype));
		if(init_) { for (int i=0; i<(my*mx); i++) m[i]=val0+i*inc; }
		
		
	}
	~s_matrix() {
		free(m);
	}
	
	void transpose() {
		numtype** tm=(numtype**)malloc(mx*sizeof(numtype*)); for (int y=0; y<mx; y++) tm[y]=(numtype*)malloc(my*sizeof(numtype));
		for (int y = 0; y < my; y++) {
			for (int x = 0; x < mx; x++) {
				tm[x][y] = m[y*mx+x];
			}
		}
		for (int y = 0; y < my; y++) {
			for (int x = 0; x < mx; x++) {
				m[x*my+y]=tm[x][y];
			}
		}

		for (int y=0; y<mx; y++) free(tm[y]);
		free(tm);

		int tmp=my;	my=mx; mx=tmp;
	}
	int transposeTo(s_matrix* otm) {
		if (otm->mx!=my||otm->my!=mx) {
			printf("transposeTo() dimensions mismatch!\n");
			return -1;
		}
		for (int y = 0; y < my; y++) {
			for (int x = 0; x < mx; x++) {
				otm->m[x*my+y] = m[y*mx+x];
			}
		}
		return 0;
	}
	void fill(numtype start, numtype inc) {
		for (int i=0; i<(my*mx); i++) m[i]=start+i*inc;
	}
	void scale(float s) {
		for (int i=0; i<(my*mx); i++) m[i]*=s;
	}
	void print(const char* msg=nullptr, int smy0=-1, int smx0=-1, int smy=-1, int smx=-1) {
		if (smy==-1) smy=my;
		if (smx==-1) smx=mx;

		int idx;
		if (msg!=nullptr) printf("%s [%dx%d] - from [%d,%d] to [%d,%d]\n", msg, my, mx, (smy0==-1)?0:smy0,(smx0==-1)?0:smx0,smy0+smy,smx0+smx);
		for (int y=0; y<smy; y++) {
			for (int x=0; x<smx; x++) {
				idx= y*mx+x;
				printf("|%4.1f", m[idx]);
			}
			printf("|\n");
		}
	}
	int copyTo(s_matrix* tom) {
		if(tom->my!=my || tom->mx!=mx) {
			printf("copyTo() can only work with same-sized matrices!\n");
			return -1;
		}
		for (int i=0; i<(my*mx); i++) tom->m[i]=m[i];
		return 0;
	}
	int copySubTo(int y0=0, int x0=0, s_matrix* osm=nullptr) {
		if (osm==nullptr) return -1;

		int idx;
		int odx=0;
		for (int y=y0; y<(y0+osm->my); y++) {
			for (int x=x0; x<(x0+osm->mx); x++) {
				idx= y*this->mx+x;
				osm->m[odx]=m[idx];
				odx++;
			}
		}

		return 0;
	}

	int X(s_matrix* B, s_matrix* C, bool trA, bool trB, float Ascale=1, float Bscale=1) {
		if (trA) swap(&mx, &my);
		if (trB) swap(&B->mx, &B->my);

		for (int y = 0; y < my; y++) {
			for (int x2 = 0; x2 < B->mx; x2++) {
				C->m[y*B->mx+x2] = 0;
				for (int x = 0; x < mx; x++) {
					C->m[y*B->mx+x2] += m[y*mx+x]*Ascale * B->m[x*B->mx+x2]*Bscale;
				}
			}
		}
		return 0;
	}
} matrix;

EXPORT void Mprint(int my, int mx, numtype* sm, const char* msg=nullptr, int smy0=-1, int smx0=-1, int smy=-1, int smx=-1);
EXPORT void Msub(int my, int mx, numtype* INm, numtype* OUTsm, int smy0, int smx0, int smy, int smx);
EXPORT int Vsum(int Vlen, int* V);

//-- TODO: CUDA VERSIONS !!!
EXPORT void Vscale(int Vlen, int* V, float s);
//--

EXPORT int Vscale(int vlen, numtype* v, numtype s);
EXPORT int Vcopy(int vlen, numtype* v1, numtype* v2);
EXPORT int Vadd(int vlen, numtype* v1, numtype scale1, numtype* v2, numtype scale2, numtype* ov);
EXPORT int Vdiff(int vlen, numtype* v1, numtype scale1, numtype* v2, numtype scale2, numtype* ov);
EXPORT int Vsum(int Vlen, numtype* V, numtype* oSum);
EXPORT int Vssum(int Vlen, numtype* V, numtype* osSum);
EXPORT int Vnorm(void* cublasH, int Vlen, numtype* V, numtype* oVnorm);
EXPORT int Vinit(int size, numtype* v, numtype start, numtype inc);
EXPORT int VinitRnd(int Vlen, numtype* V, numtype rndmin, numtype rndmax, void* cuRandH=NULL);
EXPORT int VbyV2V(int Vlen, numtype* V1, numtype* V2, numtype* oV);

EXPORT int Mtranspose_std(int* my_, int* mx_, numtype* m);
EXPORT int MbyM_std(int Ay, int Ax, numtype Ascale, bool Atr, numtype* A, int By, int Bx, numtype Bscale, bool Btr, numtype* B, numtype* C, int sAy=-1, int sAx=-1, int sAy0=-1, int sAx0=-1, int sBy=-1, int sBx=-1, int sBy0=-1, int sBx0=-1, int sCy=-1, int sCx=-1, int sCy0=-1, int sCx0=-1);
EXPORT int MbyM(void* cublasH, int Ay, int Ax, numtype Ascale, bool Atr, numtype* A, int By, int Bx, numtype Bscale, bool Btr, numtype* B, numtype* C, int sAy=-1, int sAx=-1, int sAy0=-1, int sAx0=-1, int sBy=-1, int sBx=-1, int sBy0=-1, int sBx0=-1, int sCy=-1, int sCx=-1, int sCy0=-1, int sCx0=-1);

EXPORT int myMemInit(void* cublasH, void* cuRandH);
EXPORT int myMalloc(numtype** var, int size);
EXPORT int myFree(numtype* var);

EXPORT int loadBatchData(numtype* destAddr, numtype* srcAddr, int size);

EXPORT void dumpData(int vlen, numtype* v, const char* fname);

EXPORT int Tanh(int Vlen, numtype* in, numtype* out);
EXPORT int dTanh(int Vlen, numtype* in, numtype* out);
EXPORT int Exp4(int Vlen, numtype* in, numtype* out);
EXPORT int dExp4(int Vlen, numtype* in, numtype* out);
EXPORT int Relu(int Vlen, numtype* in, numtype* out);
EXPORT int dRelu(int Vlen, numtype* in, numtype* out);
EXPORT int SoftPlus(int Vlen, numtype* in, numtype* out);
EXPORT int dSoftPlus(int Vlen, numtype* in, numtype* out);
