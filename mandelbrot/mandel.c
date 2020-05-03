/* 
 * File:   mandel.c
 * Author: Antonio Lechuga
 *
 * Created on Día 9999 de la cuarentena
 */

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

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

// main 
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
        printf("Usage: mandel <log(dim)>\n");
        exit(-1);
    }

    // initialization
    time_t t1, t2;
    double max = 2.0;
    double min = -2.0;
    int array_size = r_points * i_points;
    int num_outside = 0;
    int result;
    double dR = (max - min) / r_points;
    double dI = (max - min) / i_points;
    complex c, z_temp;

    // generating arrays
    size_t size_input = array_size * sizeof(complex);
    size_t size_output = array_size * sizeof(int);
    complex *h_input = (complex *) malloc(size_input);
    int *h_output = (int *) malloc(size_output);

    // generating input
    printf("Generating input...\n");
    for (int i = 0; i < i_points; i++) {
        for (int j = 0; j < r_points; j++) {
            double real_part = min + dR * j;
            double imag_part = max - dI * i;
            h_input[i_points * i + j] = new_complex(real_part, imag_part);
        }
    }

    // generating ouput
    t1 = time(NULL);

    printf("Generating output...\n");
    for (int i = 0; i < i_points; i++) {
        for (int j = 0; j < r_points; j++) {
            // the following loop generates a 1 if
            // number is in mandelbrot series or
            // 0 if it does not.
            c = h_input[i_points * i + j];
            z_temp.real = 0.0;
            z_temp.imag = 0.0;
            result = 1;
            for (int k = 0; k < MAX_ITER; k++) {
                square_complex(&z_temp);
                add_complex(&z_temp, &c);
                if (abs_complex(&z_temp) > 2.0) {
                    result = 0;
                    num_outside++;
                    break;
                }
            }
            h_output[i_points * i + j] = result;
        }
    }

    // timing execution
    t2 = time(NULL);
    printf("Execution time: %f sec\n", difftime(t2, t1));

    // number of points outside, area and error
    printf("The number of points outside is: %d\n", num_outside);
    float area = 4.0 * max * (double)(array_size - num_outside) / (double)(array_size);
    float error = area / (double)r_points;
    printf("Area of Mandlebrot set is: %12.8f +/- %12.8f\n", area, error);

    // generating pmg image
    printf("Generating image...\n");
    FILE *fp;
    fp = fopen("mandelbrot-fractal.pgm", "w");
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
    printf("Done!\n");
    return 0;
}