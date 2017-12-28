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
EXPORT int initCURand(void* cuRandH) {
	if (curandCreateGenerator((curandGenerator_t*)cuRandH, CURAND_RNG_PSEUDO_DEFAULT)!=CURAND_STATUS_SUCCESS) {
		//if (curandCreateGenerator((curandGenerator_t*)cuRandH, CURAND_RNG_PSEUDO_DEFAULT)!=CURAND_STATUS_SUCCESS) {
		printf("CURAND initialization error!\n");
		return -1;
	}
	/* Set seed */
	if (curandSetPseudoRandomGeneratorSeed((*(curandGenerator_t*)cuRandH), 1234ULL)!=CURAND_STATUS_SUCCESS) return -1;
	return 0;
}

EXPORT int Malloc_cu(numtype** var, int size) {
	return ((cudaMalloc(var, size*sizeof(numtype))==cudaSuccess) ? 0 : -1);
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
EXPORT void dumpData_cu(int vlen, numtype* v, char* fname) {
	numtype* hw=(numtype*)malloc(vlen*sizeof(numtype));
	if (cudaMemcpy(hw, v, vlen*sizeof(numtype), cudaMemcpyDeviceToHost)!=cudaSuccess) return;
	FILE* f=fopen(fname, "w");
	for (int i=0; i<vlen; i++) fprintf(f, "%f\n", hw[i]);
	free(hw);
	fclose(f);
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

__global__ void cuVcopy_ker(const int vlen, const numtype *v1, numtype *v2) {
	int tid = blockIdx.x * blockDim.x+threadIdx.x;
	if (tid < vlen) v2[tid] = v1[tid];
}
__global__ void cuVminusV_ker(const int vlen, const numtype *a, const numtype *b, numtype* c) {
	int tid = blockIdx.x * blockDim.x+threadIdx.x;
	if (tid < vlen) c[tid] = a[tid]-b[tid];
}
__global__ void Vscale(int vlen, numtype* v, numtype scaleM, numtype scaleP) {
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	if (i<vlen) v[i] = scaleM*v[i]+scaleP;
}
__global__ void Vinit_ker(int vlen, numtype* v, numtype val) {
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	if (i<vlen) v[i] = val;
}
EXPORT int Vcopy_cu(int vlen, numtype* v1, numtype* v2) {
	dim3 gridDim;
	dim3 blockDim;
	blockDim.x = CUDA_BLOCK_SIZE;
	gridDim.x = (vlen+blockDim.x-1)/blockDim.x;

	cuVcopy_ker<<< gridDim, blockDim>>> (vlen, v1, v2);

	return((cudaGetLastError()==cudaSuccess) ? 0 : -1);
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
EXPORT int Vinit_cu(int vlen, numtype* v, numtype val) {
	dim3 gridDim;
	dim3 blockDim;
	blockDim.x = CUDA_BLOCK_SIZE;
	gridDim.x = (vlen+blockDim.x-1)/blockDim.x;

	Vinit_ker<<< gridDim, blockDim>>> (vlen, v, val);

	return((cudaGetLastError()==cudaSuccess) ? 0 : -1);
}

EXPORT int VinitRnd_cu(int vlen, numtype* v, numtype rndmin, numtype rndmax, void* cuRandH) {
	dim3 gridDim;
	dim3 blockDim;
	blockDim.x = CUDA_BLOCK_SIZE;
	gridDim.x = (vlen+blockDim.x-1)/blockDim.x;

	//-- Generate n floats on device, with  values between 0.0 and 1.0, where 0.0 is excluded and 1.0 is included
	if(curandGenerateUniform((*(curandGenerator_t*)cuRandH), v, vlen) !=CURAND_STATUS_SUCCESS) return -1;
	//-- need to scale to rndmin<->rndmax
	Vscale<<< gridDim, blockDim>>>(vlen, v, (rndmax-rndmin), rndmax-(rndmax-rndmin)*1);

	/*/-- !!!!!!!!!!!!! REMOVE !!!!!!!!!!
	numtype* hw=(numtype*)malloc(vlen*sizeof(numtype));
	if (cudaMemcpy(hw, v, vlen*sizeof(numtype), cudaMemcpyDeviceToHost)!=cudaSuccess) return -1;
	char* fname = "C:/temp/rndw.txt";
	FILE* f=fopen(fname, "w");
	for (int i=0; i<vlen; i++) fprintf(f, "%f\n", hw[i]);
	free(hw);
	fclose(f);
	//--
	*/
	return((cudaGetLastError()==cudaSuccess) ? 0 : -1);
}

__global__ void cuTanh_ker(int vlen, numtype* in, numtype* out) {
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	if (i<vlen) out[i] = tanh(in[i]);
}
__global__ void cudTanh_ker(int vlen, numtype* in, numtype* out) {
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	if (i<vlen) out[i] = 1-tanh(in[i])*tanh(in[i]);
}
__global__ void cuExp4_ker(int vlen, numtype* in, numtype* out) {
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	if (i<vlen) out[i] = 1/(1+exp(-4*in[i]));
}
__global__ void cudExp4_ker(int vlen, numtype* in, numtype* out) {
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	if (i<vlen) out[i] = 4*exp(4*in[i])/(pow(exp(4*in[i])+1, 2));
}
__global__ void cuRelu_ker(int vlen, numtype* in, numtype* out) {
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	if (i<vlen) out[i] = ((in[i] > 0) ? 1 : 0);
}
__global__ void cudRelu_ker(int vlen, numtype* in, numtype* out) {
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	if (i<vlen) out[i] = ((in[i] > 0) ? in[i] : 0);
}
__global__ void cuSoftPlus_ker(int vlen, numtype* in, numtype* out) {
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	if (i<vlen) out[i] = log(1+exp(in[i]));
}
__global__ void cudSoftPlus_ker(int vlen, numtype* in, numtype* out) {
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	if (i<vlen) out[i] = 1/(1+exp(-in[i]));
}

EXPORT int Tanh_cu(int vlen, numtype* in, numtype* out) {
	dim3 gridDim;
	dim3 blockDim;
	blockDim.x = CUDA_BLOCK_SIZE;
	gridDim.x = (vlen+blockDim.x-1)/blockDim.x;

	cuTanh_ker<<< gridDim, blockDim>>> (vlen, in, out);

	return((cudaGetLastError()==cudaSuccess) ? 0 : -1);
}
EXPORT int dTanh_cu(int vlen, numtype* in, numtype* out) {
	dim3 gridDim;
	dim3 blockDim;
	blockDim.x = CUDA_BLOCK_SIZE;
	gridDim.x = (vlen+blockDim.x-1)/blockDim.x;

	cudTanh_ker<<< gridDim, blockDim>>> (vlen, in, out);

	return((cudaGetLastError()==cudaSuccess) ? 0 : -1);
}
EXPORT int Exp4_cu(int vlen, numtype* in, numtype* out) {
	dim3 gridDim;
	dim3 blockDim;
	blockDim.x = CUDA_BLOCK_SIZE;
	gridDim.x = (vlen+blockDim.x-1)/blockDim.x;

	cuExp4_ker<<< gridDim, blockDim>>> (vlen, in, out);

	return((cudaGetLastError()==cudaSuccess) ? 0 : -1);
}
EXPORT int dExp4_cu(int vlen, numtype* in, numtype* out) {
	dim3 gridDim;
	dim3 blockDim;
	blockDim.x = CUDA_BLOCK_SIZE;
	gridDim.x = (vlen+blockDim.x-1)/blockDim.x;

	cudExp4_ker<<< gridDim, blockDim>>> (vlen, in, out);

	return((cudaGetLastError()==cudaSuccess) ? 0 : -1);
}
EXPORT int Relu_cu(int vlen, numtype* in, numtype* out) {
	dim3 gridDim;
	dim3 blockDim;
	blockDim.x = CUDA_BLOCK_SIZE;
	gridDim.x = (vlen+blockDim.x-1)/blockDim.x;

	cuRelu_ker<<< gridDim, blockDim>>> (vlen, in, out);

	return((cudaGetLastError()==cudaSuccess) ? 0 : -1);
}
EXPORT int dRelu_cu(int vlen, numtype* in, numtype* out) {
	dim3 gridDim;
	dim3 blockDim;
	blockDim.x = CUDA_BLOCK_SIZE;
	gridDim.x = (vlen+blockDim.x-1)/blockDim.x;

	cudRelu_ker<<< gridDim, blockDim>>> (vlen, in, out);

	return((cudaGetLastError()==cudaSuccess) ? 0 : -1);
}
EXPORT int SoftPlus_cu(int vlen, numtype* in, numtype* out) {
	dim3 gridDim;
	dim3 blockDim;
	blockDim.x = CUDA_BLOCK_SIZE;
	gridDim.x = (vlen+blockDim.x-1)/blockDim.x;

	cuSoftPlus_ker<<< gridDim, blockDim>>> (vlen, in, out);

	return((cudaGetLastError()==cudaSuccess) ? 0 : -1);
}
EXPORT int dSoftPlus_cu(int vlen, numtype* in, numtype* out) {
	dim3 gridDim;
	dim3 blockDim;
	blockDim.x = CUDA_BLOCK_SIZE;
	gridDim.x = (vlen+blockDim.x-1)/blockDim.x;

	cudSoftPlus_ker<<< gridDim, blockDim>>> (vlen, in, out);

	return((cudaGetLastError()==cudaSuccess) ? 0 : -1);
}
