// compute.cu
//
// driver and kernel call

#include <stdio.h>   // for printf
#include <stdlib.h>  // for malloc
#include <complex.h> // for double complex
#include <string.h>  // for memcpy()
#include <unistd.h>  // for sleep()
#include <math.h>    // for pow()
#include <stdbool.h> // for bool

#define THREADS_PER_BLOCK 512

typedef struct Qureg
{

    int rank;
    int numRanks;

    int numQubits;
    long long int numAmpsTotal;
    long long int numAmpsPerRank;

    double complex *stateVector;
    double complex *bufferVector;
} Qureg;
 
// __global__ void compute_d (double complex *a_d, double complex *b_d, int n)
__global__ void compute_d (double complex a_d, int n)

{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < n) {
        a_d[x] = a_d[x] * 2;
	}
		
}

extern "C" void matrixMultiplication(Qureg qureg, Qureg *qureg2, int arraySize)
{
    double complex a_d;
    // Qureg *b_d;    

	// int *a_d, *b_d, *c_d;

	cudaMalloc ((void**) a_d, sizeof(double complex) * arraySize);
	// cudaMalloc ((void**) &b_d, sizeof(Qureg) * arraySize);
	// cudaMalloc ((void**) &c_d, sizeof(int) * arraySize);

    // compute_d <<< ceil((float) arraySize/THREADS_PER_BLOCK), THREADS_PER_BLOCK >>> (a_d, b_d, arraySize);
    compute_d <<< ceil((float) arraySize/THREADS_PER_BLOCK), THREADS_PER_BLOCK >>> (a_d, arraySize);
	
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		printf ("CUDA error: %s\n", cudaGetErrorString(err));
		
	cudaMemcpy (qureg.stateVector, a_d, sizeof(Qureg) * arraySize, cudaMemcpyDeviceToHost);
	cudaFree (a_d);
	// cudaFree (b_d);
	// cudaFree (c_d);
}
