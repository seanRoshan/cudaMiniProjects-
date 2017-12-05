/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>

#define TILE_WIDTH 16

__global__ void mysgemm(int m, int n, int k, const float *A, const float *B, float* C) {

	/********************************************************************
	 *
	 * Compute C = A x B
	 *   where A is a (m x k) matrix
	 *   where B is a (k x n) matrix
	 *   where C is a (m x n) matrix
	 *
	 ********************************************************************/

	// INSERT KERNEL CODE HERE

	// Allocate Shared Memory
	__shared__ float sharedMemA[TILE_WIDTH][TILE_WIDTH];
	__shared__ float sharedMemB[TILE_WIDTH][TILE_WIDTH];

	unsigned int ty = threadIdx.y;
	unsigned int tx = threadIdx.x;

	// Calculate the row index of the A element and C
	unsigned int row = (blockIdx.y*TILE_WIDTH) + ty;

	// Calculate the column index of the B element and C
	unsigned int col = (blockIdx.x*TILE_WIDTH) + tx;


	

	// Stores the result of calculation
	float Cvalue = 0;

	// Just added for readability of code
	float Avalue, Bvalue = 0;

	unsigned int tilesCount = ((k - 1) / TILE_WIDTH) + 1; // Ceil (To add the potential last tile which is not completly full)

	unsigned int  row_startPoint = row * k;

	unsigned int  tile_startPoint = 0;

	for (int i = 0; i < tilesCount; i++)
	{
		tile_startPoint = i*TILE_WIDTH;

		if ( (row < m) && ((tile_startPoint + tx) < k) ){ // Check Boundry 
			sharedMemA[ty][tx] = A[tile_startPoint + row_startPoint + tx];}
		else{
			sharedMemA[ty][tx] = 0; }// Out of Boundry

		if ( (col < n) && ((tile_startPoint + ty) < k)){
			sharedMemB[ty][tx] = B[(tile_startPoint + ty)*n + col];}
		else{
			sharedMemB[threadIdx.y][threadIdx.x] = 0;}

		__syncthreads(); // Wait until every thread finish the process of loading data into the shared memory

		for (int i = 0; i < TILE_WIDTH; i++){

			Avalue = sharedMemA[ty][i];
			Bvalue = sharedMemB[i][tx];

			Cvalue += (Avalue * Bvalue);	
		}

		__syncthreads(); // wait until all threads done their jobs
	}

		// Check boundries to see if it can be fit in c or not
		if (row < m && col < n)
		{
			C[row*n + col] = Cvalue;

		}
	
}

void basicSgemm(char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc)
{
    if ((transa != 'N') && (transa != 'n')) {
	printf("unsupported value of 'transa'\n");
    	return;
    }

    if ((transb != 'N') && (transb != 'n')) {
	printf("unsupported value of 'transb'\n");
	return;
    }

    if ((alpha - 1.0f > 1e-10) || (alpha - 1.0f < -1e-10)) {
	printf("unsupported value of alpha\n");
	return;
    }

    if ((beta - 0.0f > 1e-10) || (beta - 0.0f < -1e-10)) {
	printf("unsupported value of beta\n");
	return;
    }

    // Initialize thread block and kernel grid dimensions ---------------------

	const unsigned int BLOCK_SIZE = TILE_WIDTH; // Use 16x16 thread blocks

    //INSERT CODE HERE

	// Result is m * n 
	// x = n
	// y = m
	// We get ceil to make grid bigger than an actual matrix to avoid calculation problems at the boundy of matrix

	dim3 dimGrid((int)(ceil((float)(n) / (float)BLOCK_SIZE)), (int)(ceil((float)(m) / (float)BLOCK_SIZE)),1);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE,1);


    // Invoke CUDA kernel -----------------------------------------------------

    //INSERT CODE HERE

	mysgemm <<<dimGrid, dimBlock>>> (m, n, k, A, B, C);


}


