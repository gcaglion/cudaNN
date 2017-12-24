#include "MyCU.h"

EXPORT int initCUDA() {
	// init CUDA GPU
	if (cudaSetDevice(0)!=cudaSuccess) {
		printf("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");
		return -1;
	}
	return 0;
}
EXPORT int initCUBLAS(void* cublasH) {
	// init CUBLAS

	if (cublasCreate((cublasHandle_t*)cublasH)!=CUBLAS_STATUS_SUCCESS) {
		printf("CUBLAS initialization error!\n");
		return -1;
	}

	return 0;
}

EXPORT int Malloc_cu(numtype* var, int size) {
	return ((cudaMalloc(&var, size*sizeof(numtype))==cudaSuccess) ? 0 : -1);
}

__global__	void initGPUData_ker(float *data, int numElements, float value) {
	int tid = blockIdx.x * blockDim.x+threadIdx.x;
	if (tid < numElements) {
		data[tid] = value;
	}
}
EXPORT		void initGPUData(float *data, int numElements, float value) {
	dim3 gridDim;
	dim3 blockDim;

	blockDim.x = 1024;
	gridDim.x = (numElements+blockDim.x-1)/blockDim.x;

	initGPUData_ker<<< gridDim, blockDim>>> (data, numElements, value);
}

EXPORT int loadBatchData_cu(numtype* destAddr, numtype* srcAddr, int size) {
	return ((cudaMemcpy(destAddr, srcAddr, size, cudaMemcpyHostToDevice)==cudaSuccess) ? 0 : -1);
}
EXPORT int MbyM_cu(void* cublasH,
	int fAy, int fAx, numtype Ascale, numtype* fA, int fBy, int fBx, numtype Bscale, numtype* fB, numtype* fC,
	int sAy, int sAx, int sAy0, int sAx0,
	int sBy, int sBx, int sBy0, int sBx0,
	int sCy, int sCx, int sCy0, int sCx0
) {

	const float *alpha = &Ascale;
	const float *beta = &Bscale;

	//-- starting point and yx size for A,B,C
	int pAy=(sAy==-1) ? fAy : sAy;
	int pAx=(sAx==-1) ? fAx : sAx;
	int pA0=(sAy==-1||sAx==-1) ? 0 : (sAx0+fAx*sAy0);
	//int pBy=(sBy==-1) ? fBy : sBy;
	int pBx=(sBx==-1) ? fBx : sBx;
	int pB0=(sBy==-1||sBx==-1) ? 0 : (sBx0+fBx*sBy0);
	//int pCy=(sCy==-1) ? fCy : sCy;
	//int pCx=(sCx==-1) ? fCx : sCx;
	int fCx=fBx;

	// Do the actual multiplication

	if (cublasSgemm((*(cublasHandle_t*)cublasH), CUBLAS_OP_N, CUBLAS_OP_N, pBx, pAy, pAx, alpha, &fB[pB0], fBx, &fA[pA0], fAx, beta, fC, fCx)==CUBLAS_STATUS_SUCCESS) {
		return 0;
	} else {
		return -1;
	}

}

__global__ void cuVminusV_ker(const int vlen, const numtype *a, const numtype *b, numtype* c) {
	int tid = blockIdx.x * blockDim.x+threadIdx.x;
	if (tid < vlen) c[tid] = a[tid]-b[tid];
}
EXPORT int Vdiff_cu(int vlen, numtype* v1, numtype* v2, numtype* ov) {
	dim3 gridDim;
	dim3 blockDim;

	blockDim.x = CUDA_BLOCK_SIZE;
	gridDim.x = (vlen+blockDim.x-1)/blockDim.x;

	cuVminusV_ker<<< gridDim, blockDim>>> (vlen, v1, v2, ov);

	return((cudaGetLastError()==cudaSuccess) ? 0 : -1);
}
EXPORT int Vnorm_cu(void* cublasH, int Vlen, numtype* V,  numtype* oVnorm) {
	return ((cublasSnrm2_v2((*(cublasHandle_t*)cublasH), Vlen, V, 1, oVnorm)==CUBLAS_STATUS_SUCCESS) ? 0 : -1);
}

EXPORT __global__ void cuTanh(int Vlen, numtype* in, numtype* out) {
	int tid = blockIdx.x;
	if (tid < Vlen) out[tid] = tanh(in[tid]);
}
EXPORT __global__ void cudTanh(int Vlen, numtype* in, numtype* out) {
	int tid = blockIdx.x;
	if (tid < Vlen) out[tid] = 1-tanh(in[tid])*tanh(in[tid]);
}
EXPORT __global__ void cuExp4(int Vlen, numtype* in, numtype* out) {
	int tid = blockIdx.x;
	if (tid < Vlen) out[tid] = 1/(1+exp(-4*in[tid]));
}
EXPORT __global__ void cudExp4(int Vlen, numtype* in, numtype* out) {
	int tid = blockIdx.x;
	if (tid < Vlen) out[tid] = 4*exp(4*in[tid])/(pow(exp(4*in[tid])+1, 2));
}
EXPORT __global__ void cuRelu(int Vlen, numtype* in, numtype* out) {
	int tid = blockIdx.x;
	if (tid < Vlen) out[tid] = ((in[tid] > 0) ? 1 : 0);
}
EXPORT __global__ void cudRelu(int Vlen, numtype* in, numtype* out) {
	int tid = blockIdx.x;
	if (tid < Vlen) out[tid] = ((in[tid] > 0) ? in[tid] : 0);
}
EXPORT __global__ void cuSoftPlus(int Vlen, numtype* in, numtype* out) {
	int tid = blockIdx.x;
	if (tid < Vlen) out[tid] = log(1+exp(in[tid]));
}
EXPORT __global__ void cudSoftPlus(int Vlen, numtype* in, numtype* out) {
	int tid = blockIdx.x;
	if (tid < Vlen) out[tid] = 1/(1+exp(-in[tid]));
}
