#include "MyCU.h"

void swap(int* v1, int* v2) {
	int tmp=(*v1);
	(*v1)=(*v2);
	(*v2)=tmp;
}

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
EXPORT int Free_cu(numtype* var) {
	return (cudaFree(var));
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
EXPORT void dumpData_cu(int vlen, numtype* v, const char* fname) {
	numtype* hw=(numtype*)malloc(vlen*sizeof(numtype));
	if (cudaMemcpy(hw, v, vlen*sizeof(numtype), cudaMemcpyDeviceToHost)!=cudaSuccess) return;
	FILE* f=fopen(fname, "w");
	for (int i=0; i<vlen; i++) fprintf(f, "%f\n", hw[i]);
	free(hw);
	fclose(f);
}

//-- matrix functions
EXPORT int cuMtr_cublas(void* cublasH, int my, int mx, numtype* m, numtype* otm) {
	float alpha=1;
	float beta=0;
	if (cublasSgeam((*(cublasHandle_t*)cublasH), CUBLAS_OP_T, CUBLAS_OP_T, my, mx, &alpha, m, mx, &beta, m, mx, otm, my)!=CUBLAS_STATUS_SUCCESS) return -1;
	return 0;
}

EXPORT int MbyM_cu(void* cublasH, int Ay, int Ax, numtype Ascale, bool Atr, numtype* A, int By, int Bx, numtype Bscale, bool Btr, numtype* B, numtype* C, numtype* T) {

	float *alpha = &Ascale;
	float *beta = &Bscale;

	cublasOperation_t Aop=CUBLAS_OP_N;
	cublasOperation_t Bop=CUBLAS_OP_N;

	int Cx=(Btr) ? By : Bx;

	int m=Bx;
	int n=Ay;
	int k=Ax;
	int ldA=Ax;
	int ldB=Bx;
	int ldC=Cx;

	numtype* vA = A;
	numtype* vB = B;

	if (Atr) {
		if (cuMtr_cublas(cublasH, Ay, Ax, A, T)!=0) return -1;
		vA=T;
		n=Ax; k=Ay;
		ldA=Ay;
	}
	if (Btr) {
		if (cuMtr_cublas(cublasH, By, Bx, B, T)!=0) return -1;
		vB=T;
		m=By;
		ldB=By;
	}

	if (cublasSgemm((*(cublasHandle_t*)cublasH), Bop, Aop, m, n, k, alpha, vB, ldB, vA, ldA, beta, C, ldC)!=CUBLAS_STATUS_SUCCESS) return -1;

	return 0;
}

__global__ void cuSadd(const numtype* s1, const numtype* s2, numtype* ssum) {
	ssum[0]=s1[0]+s2[0];
}
__global__ void cuVscale_ker(const int vlen, numtype *v, const numtype s) {
	int tid = blockIdx.x * blockDim.x+threadIdx.x;
	if (tid < vlen) v[tid] *= s;
}
__global__ void cuVcopy_ker(const int vlen, const numtype *v1, numtype *v2) {
	int tid = blockIdx.x * blockDim.x+threadIdx.x;
	if (tid < vlen) v2[tid] = v1[tid];
}
__global__ void cuVminusV_ker(const int vlen, const numtype *a, const numtype sa, const numtype *b, const numtype sb, numtype* c) {
	int tid = blockIdx.x * blockDim.x+threadIdx.x;
	if (tid < vlen) c[tid] = a[tid]*sa-b[tid]*sb;
}
__global__ void cuVplusV_ker(const int vlen, const numtype *a, const numtype sa, const numtype *b, const numtype sb, numtype* c) {
	int tid = blockIdx.x * blockDim.x+threadIdx.x;
	if (tid < vlen) c[tid] = a[tid]*sa+b[tid]*sb;
}
__global__ void cuVsum_ker(const int vlen, const numtype *v, numtype* osum) {

	//@@ Load a segment of the input vector into shared memory
	__shared__ float partialSum[2*CUDA_BLOCK_SIZE];
	unsigned int t = threadIdx.x, start = 2*blockIdx.x * CUDA_BLOCK_SIZE;
	if (start+t < vlen)
		partialSum[t] = v[start+t];
	else
		partialSum[t] = 0;
	if (start+CUDA_BLOCK_SIZE+t < vlen)
		partialSum[CUDA_BLOCK_SIZE+t] = v[start+CUDA_BLOCK_SIZE+t];
	else
		partialSum[CUDA_BLOCK_SIZE+t] = 0;
	//@@ Traverse the reduction tree
	for (unsigned int stride = CUDA_BLOCK_SIZE; stride>=1; stride >>= 1) {
		__syncthreads();
		if (t < stride)
			partialSum[t] += partialSum[t+stride];
	}
	//@@ Write the computed sum of the block to the output vector at the 
	//@@ correct index
	if (t==0)
		osum[blockIdx.x] = partialSum[0];

}
__global__ void cuVssum_ker(const int vlen, const numtype *v, numtype* ossum) {

	//@@ Load a segment of the input vector into shared memory
	__shared__ float partialSum[2*CUDA_BLOCK_SIZE];
	unsigned int t = threadIdx.x, start = 2*blockIdx.x * CUDA_BLOCK_SIZE;
	if (start+t < vlen)
		partialSum[t] = v[start+t]*v[start+t];
	else
		partialSum[t] = 0;
	if (start+CUDA_BLOCK_SIZE+t < vlen)
		partialSum[CUDA_BLOCK_SIZE+t] = v[start+CUDA_BLOCK_SIZE+t]*v[start+CUDA_BLOCK_SIZE+t];
	else
		partialSum[CUDA_BLOCK_SIZE+t] = 0;
	//@@ Traverse the reduction tree
	for (unsigned int stride = CUDA_BLOCK_SIZE; stride>=1; stride >>= 1) {
		__syncthreads();
		if (t < stride)
			partialSum[t] += partialSum[t+stride];
	}
	//@@ Write the computed sum of the block to the output vector at the 
	//@@ correct index
	if (t==0)
		ossum[blockIdx.x] = partialSum[0];

}
__global__ void Vscale(int vlen, numtype* v, numtype scaleM, numtype scaleP) {
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	if (i<vlen) v[i] = scaleM*v[i]+scaleP;
}
__global__ void Vinit_ker(int vlen, numtype* v, numtype start, numtype inc) {
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	if (i<vlen) v[i] = start+i*inc;
}
__global__ void VbyV2V_ker(int vlen, numtype* v1, numtype* v2, numtype* ov) {
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	if (i<vlen) ov[i]=v1[i]*v2[i];
}

//-- scalar functions
EXPORT int Sadd_cu(numtype* s1, numtype* s2, numtype* ssum) {
	cuSadd<<< 1, 1>>>(s1, s2, ssum);
	return ((cudaGetLastError()==cudaSuccess) ? 0 : -1);
}

//-- vector functions;
EXPORT int Vscale_cu(int vlen, numtype* v, numtype s){
	dim3 gridDim;
	dim3 blockDim;
	blockDim.x = CUDA_BLOCK_SIZE;
	gridDim.x = (vlen+blockDim.x-1)/blockDim.x;

	cuVscale_ker<<< gridDim, blockDim>>> (vlen, v, s);

	return((cudaGetLastError()==cudaSuccess) ? 0 : -1);
}
EXPORT int Vcopy_cu(int vlen, numtype* v1, numtype* v2) {
	dim3 gridDim;
	dim3 blockDim;
	blockDim.x = CUDA_BLOCK_SIZE;
	gridDim.x = (vlen+blockDim.x-1)/blockDim.x;

	cuVcopy_ker<<< gridDim, blockDim>>> (vlen, v1, v2);

	return((cudaGetLastError()==cudaSuccess) ? 0 : -1);
}
EXPORT int Vadd_cu(int vlen, numtype* v1, numtype scale1, numtype* v2, numtype scale2, numtype* ov) {
	dim3 gridDim;
	dim3 blockDim;
	blockDim.x = CUDA_BLOCK_SIZE;
	gridDim.x = (vlen+blockDim.x-1)/blockDim.x;

	cuVplusV_ker<<< gridDim, blockDim>>> (vlen, v1, scale1, v2, scale2, ov);

	return((cudaGetLastError()==cudaSuccess) ? 0 : -1);
}
EXPORT int Vdiff_cu(int vlen, numtype* v1, numtype scale1, numtype* v2, numtype scale2, numtype* ov) {
	dim3 gridDim;
	dim3 blockDim;
	blockDim.x = CUDA_BLOCK_SIZE;
	gridDim.x = (vlen+blockDim.x-1)/blockDim.x;

	cuVminusV_ker<<< gridDim, blockDim>>> (vlen, v1, scale1, v2, scale2, ov);

	return((cudaGetLastError()==cudaSuccess) ? 0 : -1);
}
EXPORT int Vsum_cu(int vlen, numtype* v, numtype* ovsum, numtype* ss_d) {
	dim3 gridDim;
	dim3 blockDim;
	blockDim.x = CUDA_BLOCK_SIZE;
	gridDim.x = (vlen+blockDim.x-1)/blockDim.x;

	cuVsum_ker<<< gridDim, blockDim>>> (vlen, v, ss_d );

	if (cudaMemcpy(ovsum, ss_d, sizeof(numtype), cudaMemcpyDeviceToHost)!=cudaSuccess) return -1;

	return ((cudaGetLastError()==cudaSuccess) ? 0 : -1);
}
/*EXPORT int Vssum_cu(int vlen, numtype* v, numtype* ovssum, numtype* ss_d) {
	dim3 gridDim;
	dim3 blockDim;
	blockDim.x = CUDA_BLOCK_SIZE;
	gridDim.x = (vlen+blockDim.x-1)/blockDim.x;

	cuVssum_ker<<< gridDim, blockDim>>> (vlen, v, ss_d);

	if (cudaMemcpy(ovssum, ss_d, sizeof(numtype), cudaMemcpyDeviceToHost)!=cudaSuccess) return -1;

	return ((cudaGetLastError()==cudaSuccess) ? 0 : -1);
}
*/
EXPORT int Vssum_cu(void* cublasH, int Vlen, numtype* V, numtype* oVssum, numtype* ss_d) {
	if (cublasSnrm2((*(cublasHandle_t*)cublasH), Vlen, V, 1, oVssum)!=CUBLAS_STATUS_SUCCESS) return -1;
	(*oVssum)=(*oVssum)*(*oVssum);
	return 0;
}
EXPORT int Vnorm_cu(void* cublasH, int Vlen, numtype* V,  numtype* oVnorm, numtype* ss_d) {
	if (cublasSnrm2((*(cublasHandle_t*)cublasH), Vlen, V, 1, oVnorm)!=CUBLAS_STATUS_SUCCESS) return -1;
	if (cudaMemcpy(oVnorm, ss_d, sizeof(numtype), cudaMemcpyDeviceToHost)!=cudaSuccess) return -1;
	return 0;
}
EXPORT int Vinit_cu(int vlen, numtype* v, numtype start, numtype inc) {
	dim3 gridDim;
	dim3 blockDim;
	blockDim.x = CUDA_BLOCK_SIZE;
	gridDim.x = (vlen+blockDim.x-1)/blockDim.x;

	Vinit_ker<<< gridDim, blockDim>>> (vlen, v, start, inc);

	return((cudaGetLastError()==cudaSuccess) ? 0 : -1);
}
EXPORT int VbyV2V_cu(int vlen, numtype* v1, numtype* v2, numtype* ov) {
	dim3 gridDim;
	dim3 blockDim;
	blockDim.x = CUDA_BLOCK_SIZE;
	gridDim.x = (vlen+blockDim.x-1)/blockDim.x;

	VbyV2V_ker<<< gridDim, blockDim>>> (vlen, v1, v2, ov);

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
