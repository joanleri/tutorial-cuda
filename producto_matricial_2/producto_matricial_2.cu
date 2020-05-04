/* 
* 
* Multiplicación de Matrices en CUDA
* 
*/

#include <stdio.h>
#include <stdlib.h>
#include <cassert>
#include <time.h>
//PP#include <cuda.h>

#define SHMEM_SIZE 32 * 32 * 4   // guardar en L1 256 float's

/* Utilidad para checar errores de CUDA */
void checkCUDAError(const char*);

// Kernel de multiplicación de matrices
__global__ void matrix_multiplication(float *d_A, float *d_B, float *d_C, int N, int tile_size) {

    // memoria compartida
    __shared__ 
    float A[SHMEM_SIZE];
    __shared__ 
    float B[SHMEM_SIZE];

    // indices de hilos y bloques
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // calculando columna y file
    int row = by * tile_size + ty;
    int col = bx * tile_size + tx;

    // inticailizando suma temporal
    float temp = 0.0;

    // Realizando operaciones
    for (int i = 0; i < (N / tile_size); i++) {

        // cargando memoria compartida con porción de las "matrices"
        A[(ty * tile_size) + tx] = d_A[row * N + (i * tile_size + tx)];
        B[(ty * tile_size) + tx] = d_B[(i * tile_size * N + ty * N) + col];

        // esperando hilos para que todo esté cargado
        __syncthreads();

        // calculando temp
        for (int j = 0; j < tile_size; j++) {
            temp += A[(ty * tile_size + j)] * B[j * tile_size + tx];
        }

        // esperando hilos para evitar que se cargue nueva información antes 
        // de que todos los hilos terminen de acceder a la memoria compartida
        __syncthreads();
    }

    d_C[row * N + col] = temp;
}

// Verificando resultado en el CPU
void verify_result(float *A, float *B, float *C, int N) {
    for (unsigned int i = 0; i < N; i++) {
        for (unsigned int j = 0; j < N; j++) {
            float sum = 0;
            for (unsigned int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            // check against GPU result
            assert(sum == C[i * N + j]);
        }
    }
}

// Main routine
int main(int argc, char *argv[]) {
    float *h_A, *h_B, *h_C;                     // matrices en CPU
    float *d_A, *d_B, *d_C;                     // matrices en GPU

    if (argc < 2) {
        printf("usage: mul <matrix-dimension-power-2>\n");
        exit(-1);
    }

    if (atoi(argv[1]) < 5) {
        printf("Please provide a dimension higher than 5\n");
    }

    int N = 1 << atoi(argv[1]);                 // filas y renglones
    int MTX_SIZE = N * N;                       // matriz de tamaño
    size_t size = MTX_SIZE * sizeof(float);     // tamaño de matriz en bytes

    // Reservar memoria en CPU
    h_A = (float *) malloc(size);
    h_B = (float *) malloc(size);
    h_C = (float *) malloc(size);

    // Reservar memoria en GPU
    cudaMalloc((void **) &d_A, size);
    cudaMalloc((void **) &d_B, size);
    cudaMalloc((void **) &d_C, size);

    // inicializando matrices
    for (int i = 0; i < MTX_SIZE; i++) {
        h_A[i] = (float)(rand() % 100);
        h_B[i] = (float)(rand() % 100);
        h_C[i] = (float)0;
    }

    // copiando de CPU a GPU
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // verificando tiempo de ejecución
    time_t t1, t2;

    // corriendo kernel en el GPU
    int n_threads = 32;
    int n_blocks = N / n_threads;
    dim3 dimBlock(n_threads, n_threads);
    dim3 dimGrid(n_blocks, n_blocks);

    t1 = time(NULL);
    matrix_multiplication<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N, n_threads);
    
    // esperando a que acaben los hilos
    cudaThreadSynchronize();
    checkCUDAError("kernel invocation");

    // timing execution
    t2 = time(NULL);
    printf("Execution time: %f sec\n", difftime(t2, t1));

    // copiando resultado de regreso al CPU
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    checkCUDAError("memcpy");

    // verificando resultado
    // printf("Verifying result in CPU...\n");
    // verify_result(h_A, h_B, h_C, N);
    // printf("Success!\n");

    // Liberar memoria
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// Utility function to check for and report CUDA errors
void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }
}