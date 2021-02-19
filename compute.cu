// compute.cu
//
// driver and kernel call

#include <stdio.h>   // for printf
#include <stdlib.h>  // for malloc
#include <string.h>  // for memcpy()
#include <unistd.h>  // for sleep()
#include <math.h>    // for pow()
#include <stdbool.h> // for bool

#define THREADS_PER_BLOCK 512
 
// __global__ void compute_d (double complex *a_d, double complex *b_d, int n)
__global__ void compute_d (double *a_d, double *b_d, double *c_d, double *d_d, double *e_d, double *f_d, int n)
{

    // int x = blockIdx.x * blockDim.x + threadIdx.x;
    // int y = blockIdx.y * blockDim.y + threadIdx.y;
    // if (x < n && y < n) {
    //     // e_d[x] = e_d[x] + (a_d[y] * c_d[y + x * n]) + (b_d[y] * d_d[y + x * n] * -1);
    //     // f_d[x] = f_d[x] + (a_d[y] * d_d[y + x * n]) + (b_d[y] * c_d[y + x * n]);
        
    //     e_d[x] = e_d[x] + (a_d[y] * c_d[x + y * n]) + (b_d[y] * d_d[x + y * n] * -1);
    //     f_d[x] = f_d[x] + (a_d[y] * d_d[x + y * n]) + (b_d[y] * c_d[x + y * n]);
    // }
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < n) {

        for (int i = 0; i < n; i++)
        {
            e_d[x] = e_d[x] + (a_d[i] * c_d[x + i * n]) + (b_d[i] * d_d[x + i * n] * -1);
            f_d[x] = f_d[x] + (a_d[i] * d_d[x + i * n]) + (b_d[i] * c_d[x + i * n]);
        }
        
    }
    __syncthreads();
		
}

extern "C" void matrixMultiplication(double *quregReal, double *quregImg, double *qureg2Real, double *qureg2Img, long long arraySize)
{
    double *a_d, *b_d, *c_d, *d_d, *e_d, *f_d;

    cudaMalloc ((void**) &a_d, sizeof(double) * arraySize);
    cudaMalloc ((void**) &b_d, sizeof(double) * arraySize);
    cudaMalloc ((void**) &c_d, sizeof(double) * arraySize * arraySize);
    cudaMalloc ((void**) &d_d, sizeof(double) * arraySize * arraySize);
    cudaMalloc ((void**) &e_d, sizeof(double) * arraySize);
    cudaMalloc ((void**) &f_d, sizeof(double) * arraySize);

    cudaMemcpy (a_d, quregReal, sizeof(double) * arraySize, cudaMemcpyHostToDevice);
    cudaMemcpy (b_d, quregImg, sizeof(double) * arraySize, cudaMemcpyHostToDevice);
    cudaMemcpy (c_d, qureg2Real, sizeof(double) * arraySize * arraySize, cudaMemcpyHostToDevice);
    cudaMemcpy (d_d, qureg2Img, sizeof(double) * arraySize * arraySize, cudaMemcpyHostToDevice);
    
    compute_d <<< ceil((float) arraySize/THREADS_PER_BLOCK), THREADS_PER_BLOCK >>> (a_d, b_d, c_d, d_d, e_d, f_d, arraySize);
	
	// cudaError_t err = cudaGetLastError();
	// if (err != cudaSuccess)
	// 	printf ("CUDA error: %s\n", cudaGetErrorString(err));
		
    cudaMemcpy (quregReal, e_d, sizeof(double) * arraySize, cudaMemcpyDeviceToHost);
    cudaMemcpy (quregImg, f_d, sizeof(double) * arraySize, cudaMemcpyDeviceToHost);

	cudaFree (a_d);
    cudaFree (b_d);
    cudaFree (c_d);
	cudaFree (d_d);
}
