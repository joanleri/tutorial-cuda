/* 
* File:   mandel.c
* Author: Antonio Lechuga
*
* Created on Día 9999 de la cuarentena COVID19
*/

#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
//PP#include <cuda.h>
//PP#include <cuComplex.h>

# define POINTS_PER_DIM 1024
# define MAX_ITER 2000

/* Utilidad para checar errores de CUDA */
void checkCUDAError(const char*);

// Mandelbrot generation kernel
__global__ void generate_mandelbrot(double complex *in, int *out, int i_size, int max_iter) {
    
    // calculating indices
    int id_r = blockIdx.x * blockDim.x + threadIdx.x;
    int id_i = blockIdx.y * blockDim.y + threadIdx.y;

    // initial values
    double complex z = 0.0 + 0.0 * I;
    double complex c = in[id_i * i_size + id_r];
    int result = 1;

    // determining if c is part of mandelbrot set
    for (int i = 0; i < max_iter; i++) {
        z = z * z + c;
        if (cabs(z) > 2.0) {
            result = 0;
            break;
        }
    }
    out[id_i * i_size + id_r] = result;
}

int main(int argc, char** argv) {

    // parsing input
    int r_points, i_points;
    if (argc < 2) {
        r_points = POINTS_PER_DIM;
        i_points = POINTS_PER_DIM;
    } else if (argc < 3) {
        r_points = 1 << atoi(argv[1]);
        i_points = 1 << atoi(argv[1]);
    } else if (argc < 4) {
        r_points = 1 << atoi(argv[1]);
        i_points = 1 << atoi(argv[2]);
    } else {
        printf("Usage: mandel-gpu <log(xdim)> <log(ydim)>\n");
        exit(-1);
    }

    // initialization
    time_t t1, t2;
    double max = 2.0;
    double min = -2.0;
    int array_size = r_points * i_points;
    // int num_outside = 0;
    // int result;
    double dR = (max - min) / r_points;
    double dI = (max - min) / i_points;

    // calculating sizes
    size_t size_input = array_size * sizeof(double complex);
    size_t size_output = array_size * sizeof(int);

    // pointers
    double complex *h_input;           // CPU
    double complex *d_input;           // CPU
    int *h_output;                     // GPU
    int *d_output;                     // GPU

    // allocating space in CPU
    h_input = (double complex *) malloc(size_input);
    h_output = (int *) malloc(size_output);

    // allocating space in GPU
    cudaMalloc((void **) &d_input, size_input);
    cudaMalloc((void **) &d_output, size_output);


    // generating input
    printf("Generating input...\n");
    for (int i = 0; i < i_points; i++) {
        for (int j = 0; j < r_points; j++) {
            double real_part = min + dR * j;
            double imag_part = max - dI * i;
            h_input[i_points * i + j] = real_part + imag_part * I;
        }
    }

    // copying from CPU to GPU
    cudaMemcpy(d_input, h_input, size_input, cudaMemcpyHostToDevice);

    // executing kernels
    t1 = time(NULL);
    int n_threads = 32;
    int n_blocks_r = r_points / n_threads;
    int n_blocks_i = i_points / n_threads;
    dim3 dimBlock(n_threads, n_threads);
    dim3 dimGrid(n_blocks_r, n_blocks_i);

    generate_mandelbrot<<<dimGrid, dimBlock>>>(d_input, d_output, i_points, MAX_ITER);
    
    // waiting for threads
    cudaThreadSynchronize();
    checkCUDAError("kernel invocation");

    // timing execution
    t2 = time(NULL);
    printf("Execution time: %f sec\n", difftime(t2, t1));

    // copying back to CPU
    cudaMemcpy(h_output, d_output, size_output, cudaMemcpyDeviceToHost);
    checkCUDAError("memcpy");

    // generating pmg image
    printf("Generating image...\n");
    FILE *fp;
    fp = fopen("mandelbrot-fractal-gpu.pgm", "w");
    fputs("P2 \n", fp);
    fprintf(fp, "%d %d \n", i_points, r_points);
    fputs("1 \n", fp);
    for (int i = 0; i < i_points; i++) {
        for (int j = 0; j < r_points; j++) {
            fprintf(fp, "%d ", h_output[i * i_points + j]);
        }
        fputs("\n", fp);
    }
    fclose(fp);

    // freeing memory
    printf("Freeing memory...\n");
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    printf("Done!\n");
    return 0;
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