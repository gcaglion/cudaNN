#include "MyCU.h"

//-- CUDA (no CUBLAS) matrix utilities
#define TILE_DIM 32
#define BLOCK_ROWS 8
__global__ void transposeNaive(float *odata, const float *idata)
{
	int x = blockIdx.x * TILE_DIM+threadIdx.x;
	int y = blockIdx.y * TILE_DIM+threadIdx.y;
	int width = gridDim.x * TILE_DIM;

	for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS)
		odata[x*width+(y+j)] = idata[(y+j)*width+x];
}
__global__ void transposeCoalesced(float *odata, const float *idata) {

	__shared__ float tile[TILE_DIM][TILE_DIM];

	int x = blockIdx.x * TILE_DIM+threadIdx.x;
	int y = blockIdx.y * TILE_DIM+threadIdx.y;
	int width = gridDim.x * TILE_DIM;

	for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
		tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*width+x];

	__syncthreads();

	x = blockIdx.y * TILE_DIM+threadIdx.x;  // transpose block offset
	y = blockIdx.x * TILE_DIM+threadIdx.y;

	for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
		odata[(y+j)*width+x] = tile[threadIdx.x][threadIdx.y+j];
}
__global__ void transpose_per_element_tiled(float* t, const float* m, int matrixSize) {
	int col = blockIdx.x * blockDim.x+threadIdx.x;        // col 
	int row = blockIdx.y * blockDim.y+threadIdx.y;        // row 

	if (col>=matrixSize||row>=matrixSize)
		return;

	extern __shared__ float tile[];

	// Coalesced read from global memory - TRANSPOSED write into shared memory 

	int from = row * matrixSize+col;
	int tx   = threadIdx.y+threadIdx.x * blockDim.x;    // col 
	int ty   = threadIdx.y * blockDim.y+threadIdx.x;    // row 

	tile[ty] = m[from];
	__syncthreads();

	// Read from shared memory - coalesced write to global memory 
	int to   = (blockIdx.y * blockDim.y+threadIdx.x)+(blockIdx.x * blockDim.x+threadIdx.y) * matrixSize;

	t[to] = tile[tx];
}

EXPORT int Mtranspose_cu(int my, int mx, numtype* m, numtype* omt) {

	int matrixSize=my*mx;
	dim3 block(1024);
	dim3 grid(matrixSize/block.x);

	// 6. Like #5, but reduced block dimension from 1024 threads to 256 (16x16) 
	//    Transpose with thread per element tiled and padded to avoid bank conflicts 
	//    We pad y dimension by one element  
	block.x = 16;
	block.y = 16;
	grid.x = (matrixSize/block.x);
	grid.y = (matrixSize/block.y);

	transpose_per_element_tiled<<< grid, block, block.x * (block.y+1)*sizeof(float)>>>(omt, m, matrixSize);

	cudaDeviceSynchronize();
	int err=cudaGetLastError();
	return((cudaGetLastError()==cudaSuccess) ? 0 : -1);

}
//--

__global__ void copy(float *odata, float* idata, int width, int height, int nreps) {
	int xIndex = blockIdx.x*TILE_DIM+threadIdx.x;
	int yIndex = blockIdx.y*TILE_DIM+threadIdx.y;
	int index = xIndex+width*yIndex; 
	for (int r=0; r < nreps; r++) {
		for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
			odata[index+i*width] = idata[index+i*width];
		}
	}
}
__global__ void transposeNaive(float *odata, float* idata, int width, int height, int nreps) {
	int xIndex = blockIdx.x*TILE_DIM+threadIdx.x;
	int yIndex = blockIdx.y*TILE_DIM+threadIdx.y;
	int index_in = xIndex+width * yIndex;
	int index_out = yIndex+height * xIndex;
	for (int r=0; r < nreps; r++) {
		for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
			odata[index_out+i] = idata[index_in+i*width];
		}
	}
}
__global__ void transposeCoalesced(float *odata, float *idata, int width, int height, int nreps) {
	__shared__ float tile[TILE_DIM][TILE_DIM];
	int xIndex = blockIdx.x*TILE_DIM+threadIdx.x;
	int yIndex = blockIdx.y*TILE_DIM+threadIdx.y;
	int index_in = xIndex+(yIndex)*width;
	xIndex = blockIdx.y * TILE_DIM+threadIdx.x;
	yIndex = blockIdx.x * TILE_DIM+threadIdx.y;
	int index_out = xIndex+(yIndex)*height;
	for (int r=0; r < nreps; r++) {
		for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
			tile[threadIdx.y+i][threadIdx.x] =
				idata[index_in+i*width];
		}

		__syncthreads();

		for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
			odata[index_out+i*height] =
				tile[threadIdx.x][threadIdx.y+i];
		}
	}
}
__global__ void copySharedMem(float *odata, float *idata, int width, int height, int nreps)
{
	__shared__ float tile[TILE_DIM][TILE_DIM];
	int xIndex = blockIdx.x*TILE_DIM+threadIdx.x;
	int yIndex = blockIdx.y*TILE_DIM+threadIdx.y;

	int index = xIndex+width*yIndex;
	for (int r=0; r < nreps; r++) {
		for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
			tile[threadIdx.y+i][threadIdx.x] =
				idata[index+i*width];
		}

		__syncthreads();

		for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
			odata[index+i*width] =
				tile[threadIdx.y+i][threadIdx.x];
		}
	}
}
__global__ void transposeFineGrained(float *odata, float *idata, int width, int height, int nreps)
{
	__shared__ float block[TILE_DIM][TILE_DIM+1];
	int xIndex = blockIdx.x * TILE_DIM+threadIdx.x;
	int yIndex = blockIdx.y * TILE_DIM+threadIdx.y;
	int index = xIndex+(yIndex)*width;
	for (int r=0; r<nreps; r++) {
		for (int i=0; i < TILE_DIM; i += BLOCK_ROWS) {
			block[threadIdx.y+i][threadIdx.x] =
				idata[index+i*width];
		}

		__syncthreads();
		for (int i=0; i < TILE_DIM; i += BLOCK_ROWS) {
			odata[index+i*height] =
				block[threadIdx.x][threadIdx.y+i];
		}
	}
}
__global__ void transposeCoarseGrained(float *odata, float *idata, int width, int height, int nreps)
{
	__shared__ float block[TILE_DIM][TILE_DIM+1];
	int xIndex = blockIdx.x * TILE_DIM+threadIdx.x;
	int yIndex = blockIdx.y * TILE_DIM+threadIdx.y;
	int index_in = xIndex+(yIndex)*width;
	xIndex = blockIdx.y * TILE_DIM+threadIdx.x;
	yIndex = blockIdx.x * TILE_DIM+threadIdx.y;
	int index_out = xIndex+(yIndex)*height;
	for (int r=0; r<nreps; r++) {
		for (int i=0; i<TILE_DIM; i += BLOCK_ROWS) {
			block[threadIdx.y+i][threadIdx.x] =
				idata[index_in+i*width];
		}

		__syncthreads();
		for (int i=0; i<TILE_DIM; i += BLOCK_ROWS) {
			odata[index_out+i*height] =
				block[threadIdx.y+i][threadIdx.x];
		}
	}
}
__global__ void transposeDiagonal(float *odata,	float *idata, int width, int height, int nreps)
{
	__shared__ float tile[TILE_DIM][TILE_DIM+1];
	int blockIdx_x, blockIdx_y;
	// diagonal reordering
	if (width==height) {
		blockIdx_y = blockIdx.x;
		blockIdx_x = (blockIdx.x+blockIdx.y)%gridDim.x;
	} else {
		int bid = blockIdx.x+gridDim.x*blockIdx.y;
		blockIdx_y = bid%gridDim.y;
		blockIdx_x = ((bid/gridDim.y)+blockIdx_y)%gridDim.x;
	}
	int xIndex = blockIdx_x*TILE_DIM+threadIdx.x;
	int yIndex = blockIdx_y*TILE_DIM+threadIdx.y;
	int index_in = xIndex+(yIndex)*width;
	xIndex = blockIdx_y*TILE_DIM+threadIdx.x;
	yIndex = blockIdx_x*TILE_DIM+threadIdx.y;
	int index_out = xIndex+(yIndex)*height;
	for (int r=0; r < nreps; r++) {
		for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
			tile[threadIdx.y+i][threadIdx.x] =
				idata[index_in+i*width];
		}

		__syncthreads();

		for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
			odata[index_out+i*height] =
				tile[threadIdx.x][threadIdx.y+i];
		}
	}
}

#define BLOCK_DIM 16

// This kernel is optimized to ensure all global reads and writes are coalesced,
// and to avoid bank conflicts in shared memory.  This kernel is up to 11x faster
// than the naive kernel below.  Note that the shared memory array is sized to 
// (BLOCK_DIM+1)*BLOCK_DIM.  This pads each row of the 2D block in shared memory 
// so that bank conflicts do not occur when threads address the array column-wise.
__global__ void transpose(float *odata, float *idata, int width, int height) {
	__shared__ float block[BLOCK_DIM][BLOCK_DIM+1];

	// read the matrix tile into shared memory
	// load one element per thread from device memory (idata) and store it
	// in transposed order in block[][]
	unsigned int xIndex = blockIdx.x * BLOCK_DIM+threadIdx.x;
	unsigned int yIndex = blockIdx.y * BLOCK_DIM+threadIdx.y;
	if ((xIndex < width)&&(yIndex < height))
	{
		unsigned int index_in = yIndex * width+xIndex;
		block[threadIdx.y][threadIdx.x] = idata[index_in];
	}

	// synchronise to ensure all writes to block[][] have completed
	__syncthreads();

	// write the transposed matrix tile to global memory (odata) in linear order
	xIndex = blockIdx.y * BLOCK_DIM+threadIdx.x;
	yIndex = blockIdx.x * BLOCK_DIM+threadIdx.y;
	if ((xIndex < height)&&(yIndex < width))
	{
		unsigned int index_out = yIndex * height+xIndex;
		odata[index_out] = block[threadIdx.x][threadIdx.y];
	}
}

// This naive transpose kernel suffers from completely non-coalesced writes.
// It can be up to 10x slower than the kernel above for large matrices.
__global__ void transpose_naive(float *odata, float* idata, int width, int height) {
	unsigned int xIndex = blockDim.x * blockIdx.x+threadIdx.x;
	unsigned int yIndex = blockDim.y * blockIdx.y+threadIdx.y;

	if (xIndex < width && yIndex < height)
	{
		unsigned int index_in  = xIndex+width * yIndex;
		unsigned int index_out = yIndex+height * xIndex;
		odata[index_out] = idata[index_in];
	}
}

EXPORT int cuMtr_naive(int my, int mx, numtype* m, numtype* omt) {

	int size_x=mx, size_y=my;

	dim3 grid(size_x/TILE_DIM, size_y/TILE_DIM), threads(TILE_DIM, BLOCK_ROWS);
	transposeNaive<<<grid, threads>>>(omt, m, size_x, size_y, 1);

	return((cudaGetLastError()==cudaSuccess) ? 0 : 1);

}

// Number of repetitions used for timing.
#define NUM_REPS 100
 int cuMtr() {

	// set matrix size
	const int size_x = 2048, size_y = 4096;
	// kernel pointer and descriptor
	void(*kernel)(float *, float *, int, int, int);
	char *kernelName;
	// execution configuration parameters
	dim3 grid(size_x/TILE_DIM, size_y/TILE_DIM),
		threads(TILE_DIM, BLOCK_ROWS);
	// CUDA events
	cudaEvent_t start, stop;
	// size of memory required to store the matrix
	const int mem_size = sizeof(float) * size_x*size_y;
	// allocate host memory
	float *h_idata = (float*)malloc(mem_size);
	float *h_odata = (float*)malloc(mem_size);
	float *transposeGold = (float *)malloc(mem_size);
	float *gold;
	// allocate device memory
	float *d_idata, *d_odata;
	cudaMalloc((void**)&d_idata, mem_size);
	cudaMalloc((void**)&d_odata, mem_size);
	// initalize host data
	for (int i = 0; i<(size_x*size_y); ++i)
		h_idata[i] = (float)i;

	// copy host data to device
	cudaMemcpy(d_idata, h_idata, mem_size,
		cudaMemcpyHostToDevice);
/*
	// Compute reference transpose solution
	computeTransposeGold(transposeGold, h_idata, size_x, size_y);
	// print out common data for all kernels
	printf("\nMatrix size: %dx%d, tile: %dx%d, block: %dx%d\n\n",
		size_x, size_y, TILE_DIM, TILE_DIM, TILE_DIM, BLOCK_ROWS);
*/
	printf("Kernel\t\t\tLoop over kernel\tLoop within kernel\n");
	printf("------\t\t\t----------------\t------------------\n");
	//
	// loop over different kernels
	//
	for (int k = 0; k<8; k++) {
		// set kernel pointer
		switch (k) {
		case 0:
			kernel = &copy;
			kernelName = "simple copy "; break;
		case 1:
			kernel = &copySharedMem;
			kernelName = "shared memory copy "; break;
		case 2:
			kernel = &transposeNaive;
			kernelName = "naive transpose "; break;
		case 3:
			kernel = &transposeCoalesced;
			kernelName = "coalesced transpose "; break;
		case 4:
			//kernel = &transposeNoBankConflicts;
			//kernelName = "no bank conflict trans"; 
			break;
		case 5:
			kernel = &transposeCoarseGrained;
			kernelName = "coarse-grained "; break;
		case 6:
			kernel = &transposeFineGrained;
			kernelName = "fine-grained "; break;
		case 7:
			kernel = &transposeDiagonal;
			kernelName = "diagonal transpose "; break;
		}
		// set reference solution
		// NB: fine- and coarse-grained kernels are not full
		// transposes, so bypass check
		if (kernel==&copy||kernel==&copySharedMem) {
			gold = h_idata;
		} else if (kernel==&transposeCoarseGrained||
			kernel==&transposeFineGrained) {
			gold = h_odata;
		} else {
			gold = transposeGold;
		}

		// initialize events, EC parameters
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		// warmup to avoid timing startup 
		kernel<<<grid, threads>>>(d_odata, d_idata, size_x, size_y, 1);
		// take measurements for loop over kernel launches
		cudaEventRecord(start, 0);
		for (int i=0; i<NUM_REPS; i++) {
			kernel<<<grid, threads>>>(d_odata, d_idata, size_x, size_y, 1);
		}
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		float outerTime;
		cudaEventElapsedTime(&outerTime, start, stop);
		cudaMemcpy(h_odata, d_odata, mem_size, cudaMemcpyDeviceToHost);
//		int res = comparef(gold, h_odata, size_x*size_y);
//		if (res!=1)	printf("*** %s kernel FAILED ***\n", kernelName);
		// take measurements for loop inside kernel
		cudaEventRecord(start, 0);
		kernel<<<grid, threads>>>
			(d_odata, d_idata, size_x, size_y, NUM_REPS);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		float innerTime;
		cudaEventElapsedTime(&innerTime, start, stop);
		cudaMemcpy(h_odata, d_odata, mem_size, cudaMemcpyDeviceToHost);
//		res = comparef(gold, h_odata, size_x*size_y);
		//if (res!=1) printf("*** %s kernel FAILED ***\n", kernelName);

		// report effective bandwidths
		float outerBandwidth =
			2.*1000*mem_size/(1024*1024*1024)/(outerTime/NUM_REPS);
		float innerBandwidth =
			2.*1000*mem_size/(1024*1024*1024)/(innerTime/NUM_REPS);
		printf("%s\t%5.2f GB/s\t\t%5.2f GB/s\n",
			kernelName, outerBandwidth, innerBandwidth);
	}

	// cleanup
	free(h_idata); free(h_odata); free(transposeGold);
	cudaFree(d_idata); cudaFree(d_odata);
	cudaEventDestroy(start); cudaEventDestroy(stop);

	return 0;}