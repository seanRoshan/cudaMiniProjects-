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

	// Calculate the row index of the A element and C
	int row = (blockIdx.y*TILE_WIDTH) + threadIdx.y;

	// Calculate the column index of the B element and C
	int col = (blockIdx.x*TILE_WIDTH) + threadIdx.x;

	// Stores the result of calculation
	float Cvalue = 0;

	// Just added for readability of code
	float Avalue, Bvalue = 0;


	// I just restored the variable k in a variable named offset for readabiliy of my program
	// we should sum the producat of k elements in row of A, and k elements in col of B, Simple :)
	int offset = k;


	int startPoint = 0; // Start of the row
	int position = 0;  // Position of an element

	// Check if tile exceed the matrix boundry
	if ( (row<m) && (col<n)) {
	
		for (int index = 0; index < offset; index++) {
			// each thread in each blocks, compute on element of the C matrix 

			

			// Matrix A (m*k)
			// We have 1-D memory which is row major, so we should multiply row number by the number of elements in each row which is number of columns (row*k) 
			startPoint = row * k; // We should pass constant rows (row) with k elements to reach out goal 
			position = startPoint + index; // we should move inside the specific row by using index position 

			Avalue = A[position];

			// Matrix B (k*n)
			// We have 1-D memory which is row major, so we should multiply row number by the number of elements in each row which is number of columns (row*n) 
			startPoint = index * n; // We should move down index row with n elements in each row 
			position = startPoint + col; // We should stop at the constant column (col)

			Bvalue = B[position];

			Cvalue = Cvalue + (Avalue * Bvalue); // Floating point operations

			startPoint = row * n; // C matrix is m*n, so in each row it has n elements
			position = startPoint + col; // we should move left by using col to reach the goal element
			C[position] = Cvalue; // store the value in the result matrix C

		}
	
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


