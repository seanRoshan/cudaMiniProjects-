/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#define BLOCK_SIZE 1024

//Calculate and Get Histogram

__global__ void getHisto(unsigned int* input, unsigned int* bins, unsigned int num_elements,unsigned int num_bins)
{
	// Defile Dynamic Shared Memory
	// Number of bins are variable so we need dynamic shared Memory

	extern __shared__ unsigned int sharedMem[];

	// global thread index	

	int i = blockIdx.x * blockDim.x + threadIdx.x; 

	int startPoint,pos = 0;

	// Initialize the shared memory with zero value
	// In addition phase, we use old value of each element, so we should make sure that they're zero
	// before calculation
	for(int j=0; threadIdx.x+(j*BLOCK_SIZE)<num_bins; j++) {
		startPoint = j*BLOCK_SIZE;
		pos = startPoint + threadIdx.x;
		sharedMem[pos] = 0;
	        __syncthreads();
	}
	
	// Calculate local histogram
	if (i<num_elements) 
	{
		atomicAdd(&(sharedMem[input[i]]),1);	
	}
	__syncthreads();
	
   	
	// Update gpuBins 
	for(int j=0; threadIdx.x+j*BLOCK_SIZE<num_bins; j++) {
		startPoint = j*BLOCK_SIZE;
		pos = startPoint + threadIdx.x;
		atomicAdd(&(bins[pos]),sharedMem[pos]);
	}
	
}

//Convert uint32_t to unit8_t to fit into the global bins
//The problem was that atomicAdd should use 32bit operands
__global__ void convert_32_8 (uint8_t* bins, unsigned int* gpuBins, unsigned int num_bins)
{
	int i = blockIdx.x*blockDim.x +threadIdx.x;

	if(i<num_bins)
	{
		if(gpuBins[i]<=255) 
			bins[i]= (uint8_t) gpuBins[i]; // static cast 8 bit unsigned integer
		else 
			bins[i]=255;
	}
}

/******************************************************************************
 * End of Kernel Function Definitions; proceeding to the invocation section
*******************************************************************************/

void histogram(unsigned int* input, uint8_t* bins, unsigned int num_elements,unsigned int num_bins) {

	int gridSize = ((num_elements-1)/BLOCK_SIZE)+1; // number of blocks in a 1d grid

	// Allocate Dynamic Shared Memory to store histogram on GPU

	unsigned int sharedMemSize = num_bins * sizeof(unsigned int);

	// AtomicAdd does not accept uint8_t, so we need 32bit version (uint32_t or unsigned int) and initialize it to zero
	unsigned int* gpuBins;
	cudaMalloc((void**)&gpuBins,num_bins*sizeof(unsigned int));
	cudaMemset(gpuBins,0,num_bins*sizeof(unsigned int));	

	//GetHisto Kernel invocation
	getHisto<<<gridSize,BLOCK_SIZE,sharedMemSize>>>(input,gpuBins,num_elements,num_bins); 

	//cudaDeviceSynchronize();
	convert_32_8<<<gridSize,BLOCK_SIZE>>>(bins,gpuBins,num_bins);	
	
	//Free Memory
	cudaFree(gpuBins);

}