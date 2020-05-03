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


/* Utilidad para checar errores de CUDA */
void checkCUDAError(const char*);

// Kernel de multiplicación de matrices
__global__ void matrix_multiplication(float *d_A, float *d_B, float *d_C, int N) {
    // calculando renglón y columna
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // producto punto entre renglón de A y columna de B
    d_C[row * N + col] = (float)0;
    for (int i = 0; i < N; i++) {
        d_C[row * N + col] += d_A[row * N + i] * d_B[i * N + col];
    }
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
        h_A[i] = (float)i;
        h_B[i] = (float)i;
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
    matrix_multiplication<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
    

    // esperando a que acaben los hilos
    cudaThreadSynchronize();
    checkCUDAError("kernel invocation");

    // timing execution
    t2 = time(NULL);
    printf("Execution time: %f sec\n", difftime(t2, t1));

    // copiando resultado de regreso al CPU
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    checkCUDAError("memcpy");

    // // imprimiendo resultados
    // for (int i = 0; i < N; i++) {
    //     for (int j = 0; j < N; j++) {
    //         printf("%f, ", h_C[i * N + j]);
    //     }
    //     printf("\n");
    // }

    // verificando resultado
    printf("Verifying result in CPU...\n");
    verify_result(h_A, h_B, h_C, N);
    printf("Success!\n");

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