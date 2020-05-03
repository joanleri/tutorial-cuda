/* 
* File:   mandel.c
* Author: Antonio Lechuga
*
* Created on Día 9999 de la cuarentena COVID19
*/

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
//PP#include <cuda.h>

# define POINTS_PER_DIM 1024
# define MAX_ITER 2000

// Defining complex type
typedef struct complex_ {
    double real;
    double imag;
} complex, *Pcomplex;

// Getting new complex number
complex new_complex(double real, double imag) {
    Pcomplex complex_ptr = (Pcomplex)malloc(sizeof(complex));
    complex_ptr->real = real;
    complex_ptr->imag = imag;
    return *complex_ptr;
}

// Squaring complex number
// Does it with pointers so it does not need to reassign value
void square_complex(Pcomplex num) {
    float temp_real = num->real;
    num->real = (num->real * num->real) - (num->imag * num->imag);
    num->imag = 2 * temp_real * num->imag;
}

// Adding to complex number
// Does it with pointers so it does not need to reassign value
// num1 is modify automatically
void add_complex(Pcomplex num1, Pcomplex num2) {
    num1->real = num1->real + num2->real;
    num1->imag = num1->imag + num2->imag;
}

// Abs of complex number
double abs_complex(Pcomplex num) {
    return sqrt((num->real * num->real) + (num->imag * num->imag));
}


/* Utilidad para checar errores de CUDA */
void checkCUDAError(const char*);

// Mandelbrot generation kernel
__global__ void generate_mandelbrot(complex *in, int *out, int i_size, int max_iter) {
    
    // calculating indices
    int id_r = blockIdx.x * blockDim.x + threadIdx.x;
    int id_i = blockIdx.y * blockDim.y + threadIdx.y;

    // initial values
    complex z = new_complex(0.0, 0.0);
    complex c = in[id_i * i_size + id_r];
    int result = 1;

    // determining if c is part of mandelbrot set
    for (int i = 0; i < max_iter; i++) {
        square_complex(&z);
        add_complex(&z, &c);
        if (abs_complex(&z) > 2.0) {
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
    double dR = (max - min) / r_points;
    double dI = (max - min) / i_points;

    // calculating sizes
    size_t size_input = array_size * sizeof(complex);
    size_t size_output = array_size * sizeof(int);

    // pointers
    complex *h_input;                   // CPU
    complex *d_input;                   // CPU
    int *h_output;                      // GPU
    int *d_output;                      // GPU

    // allocating space in CPU
    h_input = (complex *) malloc(size_input);
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
            h_input[i_points * i + j] = new_complex(real_part, imag_part);
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