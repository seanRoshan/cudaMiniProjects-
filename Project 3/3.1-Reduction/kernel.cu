/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#define BLOCK_SIZE 512

__global__ void reduction(float *out, float *in, unsigned size)
{
    /********************************************************************
    Load a segment of the input vector into shared memory
    Traverse the reduction tree
    Write the computed sum to the output vector at the correct index
    ********************************************************************/

    // INSERT KERNEL CODE HERE


	// LOAD A SEGMENT OF THE INPUT VECTOR INTO THE SHARED MEMORY

	__shared__ float partialSum[2 * BLOCK_SIZE];

	unsigned int t = threadIdx.x;

	unsigned int start = 2*blockIdx.x*blockDim.x;

    // Each thread in a block is responsible for loading 2 elements into the shared memory 
    // Load the first element from the input 
	if (t<size){
		partialSum[t] = in[start + t];	
	}
	else {
		partialSum[t] = 0;
	}

	// Load the second element from the input (BlockSize element from the first element)

	if (blockDim.x+t<size){
		partialSum[blockDim.x+t] = in[start+ blockDim.x+t];
	}
	else {
		partialSum[blockDim.x+t] = 0;
	}

	// TRAVERSE THE REDUCTION TREE


	for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2)
	{
		// To make sure that every thread do it's job before moving into the next phase
        // On the first iteration data should be loaded 
        // On lator iterations, we need data from other threads too 
        // For Example: if t=0, stride=2, we need to add partialSum[0],partialSum[2] 
        // Thread 1 will create partialSum[2], so we should syncthreads :)

		__syncthreads();

		// If we consider block size of 4
		// Stride = 1, Active Threads: 0,1,2,3 : 4 avtive threads
		// Stride = 2, Active Threads: 0,2     : 2 active threads
		// Stride = 4, Active Threads: 0       : 1 active threads

		if (t % stride == 0) 
			partialSum[2*t]+= partialSum[2*t+stride];
	}

	// The result will be store in partialSum[t]
	// Thread 0 is the only active thread in the last iteration
	// So, it is responsible for updateing the result
	if(t == 0) {
		out[blockIdx.x + t] = partialSum[t];
	}


}
























}
