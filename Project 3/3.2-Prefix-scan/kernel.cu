/******************************************************************************
*cr
*cr            (C) Copyright 2010 The Board of Trustees of the
*cr                        University of Illinois
*cr                         All Rights Reserved
*cr
******************************************************************************/

#define BLOCK_SIZE 512

// Define your kernels in this file you may use more than one kernel if you
// need to

// INSERT KERNEL(S) HERE

/******************************************************************************
Setup and invoke your kernel(s) in this function. You may also allocate more
GPU memory if you need to
*******************************************************************************/

__global__ void preScan(float *out, float *in, unsigned int in_size){

	
	// ALLOCATE THE SHARED MEMORY

	__shared__ float XY[BLOCK_SIZE];

	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	// LOAD A SEGMENT OF THE INPUT VECTOR INTO THE SHARED MEMORY

	if (i < in_size){
		XY[threadIdx.x] = in[i];
	}

	for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {

		__syncthreads();

		int index = (threadIdx.x + 1) * 2 * stride - 1;

		if (index < blockDim.x){
			XY[index] += XY[index - stride];
		}

	}

	for (int stride = BLOCK_SIZE / 4; stride > 0; stride /= 2) {

		__syncthreads();

		int index = (threadIdx.x + 1)*stride * 2 - 1;

		if (index + stride < BLOCK_SIZE) {
			XY[index + stride] += XY[index];
		}
	}

	__syncthreads();

	out[i] = XY[threadIdx.x];
}