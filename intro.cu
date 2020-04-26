/*
 * 
 * Programa de Introducción a los conceptos de CUDA
 * 
 *
 * 
 * 
 */

#include <stdio.h>
#include <stdlib.h>

/* Declaración de métodos/


/* Utilidad para checar errores de CUDA */
void checkCUDAError(const char*);

/* Kernel para sumar dos vectores en un sólo bloque de hilos */
__global__ void vect_add(int *d_a, int *d_b, int *d_c)
{
    /* Part 2B: Implementación del kernel para realizar la suma de los vectores en el GPU */
    int idx = threadIdx.x;
    d_c[idx] = d_a[idx] + d_b[idx];
}

/* Versión de múltiples bloques de la suma de vectores */
__global__ void vect_add_multiblock(int *d_a, int *d_b, int *d_c)
{
    /* Part 2C: Implementación del kernel pero esta vez permitiendo múltiples bloques de hilos. */
    int idx = threadIdx.x + (blockIdx.x * blockDim.x);
    d_c[idx] = d_a[idx] + d_b[idx];
}

/* Numero de elementos en el vector */
#define ARRAY_SIZE 256

/*
 * Número de bloques e hilos
 * Su producto siempre debe ser el tamaño del vector (arreglo).
 */
#define NUM_BLOCKS  4
#define THREADS_PER_BLOCK 256

/* Main routine */
int main(int argc, char *argv[])
{
    int *a, *b, *c; /* Arreglos del CPU */
    int *d_a, *d_b, *d_c;/* Arreglos del GPU */

    int i;
    size_t sz = ARRAY_SIZE * sizeof(int);

    /*
     * Reservar memoria en el cpu
     */
    a = (int *) malloc(sz);
    b = (int *) malloc(sz);
    c = (int *) malloc(sz);

    /*
     * Parte 1A:Reservar memoria en el GPU
     */
    cudaMalloc((void**) &d_a, sz);
    cudaMalloc((void**) &d_b, sz);
    cudaMalloc((void**) &d_c, sz);

    /* inicialización */
    for (i = 0; i < ARRAY_SIZE; i++) {
        a[i] = i;
        b[i] = ARRAY_SIZE - i;
        c[i] = 0;
    }

    /* Parte 1B: Copiar los vectores del CPU al GPU */
    cudaMemcpy(d_a, a, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sz, cudaMemcpyHostToDevice);

    /* run the kernel on the GPU */
    /* Parte 2A: Configurar y llamar los kernels */
    dim3 dimGrid(NUM_BLOCKS, 1, 1);
    dim3 dimBlock(THREADS_PER_BLOCK / NUM_BLOCKS, 1, 1);
    //vect_add<<<dimGrid, dimBlock>>>(d_a, d_b, d_c);
    vect_add_multiblock<<<dimGrid, dimBlock>>>(d_a, d_b, d_c);

    /* Esperar a que todos los threads acaben y checar por errores */
    cudaThreadSynchronize();
    checkCUDAError("kernel invocation");

    /* Part 1C: copiar el resultado de nuevo al CPU */
    cudaMemcpy(c, d_c, sz, cudaMemcpyDeviceToHost);

    checkCUDAError("memcpy");

    /* print out the result */
    printf("Results: ");
    for (i = 0; i < ARRAY_SIZE; i++) {
      printf("%d, ", c[i]);
    }
    printf("\n\n");

    /* Parte 1D: Liberar los arreglos */
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(a);
    free(b);
    free(c);

    return 0;
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
