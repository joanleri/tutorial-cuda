/*
 *  Ejercicio 2 Práctica 4: CUDA
 *  Desempeño en función de la homogeneidad para acceder a memoria
 *  y de la regularidad del código
 */

#include <stdio.h>
//PP#include <cuda.h>

#define STRIDE       32
#define OFFSET        0
#define GROUP_SIZE  512

/* Utilidad para checar errores de CUDA */
void checkCUDAError(const char*);

// Kernel that executes on the CUDA device
__global__ void square_array(float *a, int N)
{
	int n_elem_per_thread = N / (gridDim.x * blockDim.x);
	
	// Printing the number of elements per thread using thread 0
	if (threadIdx.x == 0) {
		printf("Elements per thread: %d", n_elem_per_thread);
	}

	int block_start_idx = n_elem_per_thread * blockIdx.x * blockDim.x;
	int thread_start_idx = block_start_idx
	+ (threadIdx.x / STRIDE) * n_elem_per_thread * STRIDE
	+ ((threadIdx.x + OFFSET) % STRIDE);
	int thread_end_idx = thread_start_idx + n_elem_per_thread * STRIDE;
	if(thread_end_idx > N) thread_end_idx = N;
	int group = (threadIdx.x / GROUP_SIZE) & 1;
	for(int idx=thread_start_idx; idx < thread_end_idx; idx+=STRIDE)
	{
		if(!group) a[idx] = a[idx] * a[idx];
		else       a[idx] = a[idx] + a[idx];
	}
}

// main routine that executes on the host
int main(void)
{
	float *a_h, *a_d;  // Pointer to host & device arrays
	const int N = 1<<10;  // Make a big array with 2**10 elements
	size_t size = N * sizeof(float);
    
    /* Auxiliares para medir tiempos */
    cudaEvent_t start, stop;
    float time;

    a_h = (float *)malloc(size);        // Allocate array on host
	cudaMalloc((void **) &a_d, size);   // Allocate array on device
	
    // Initialize host array and copy it to CUDA device
	for (int i=0; i<N; i++)
        a_h[i] = (float)i;

	cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);
    checkCUDAError("memcpy");

	// Create timer for timing CUDA calculation
	//PPunsigned int timer = 0;
	//PPcutCreateTimer( &timer );
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
    
    // Set number of threads and blocks
	int n_threads_per_block = 128;//1<<9;  // 512 threads per block
	int n_blocks = 256;//1<<10;  // 1024 blocks

	// Do calculation on device

    cudaEventRecord(start,0);
	square_array <<< n_blocks, n_threads_per_block >>> (a_d, N);
	cudaThreadSynchronize();  // Wait for square_array to finish on CUDA

    checkCUDAError("kernel invocation");


	// Retrieve result from device and store it in host array
	cudaMemcpy(a_h, a_d, size, cudaMemcpyDeviceToHost);
    checkCUDAError("memcpy");

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime( &time, start, stop );

	// Print some of the results
	for (int i=0; i<N; i+=N/50) printf("%d %f\n", i, a_h[i]);

    // Imprime tiempo de ejecución
    printf("\n\nTIEMPO DE EJECUCIÓN: %f mSeg\n\n", time);

    cudaEventDestroy( start );
    cudaEventDestroy( stop );

	free(a_h); cudaFree(a_d);
}

/* Utility function to check for and report CUDA errors */
void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }
}