/* 
 * File:   mandel.c
 * Author: Antonio Lechuga
 *
 * Created on Día 9999 de la cuarentena
 */

#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

# define POINTS_PER_DIM 1024
# define MAX_ITER 2000

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
        printf("Usage: mandel <log(xdim)> <log(ydim)>\n");
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
    double complex c, z_temp;

    // generating arrays
    size_t size_input = array_size * sizeof(double complex);
    size_t size_output = array_size * sizeof(int);
    double complex *h_input = (double complex *) malloc(size_input);
    int *h_output = (int *) malloc(size_output);

    // generating input
    printf("Generating input...\n");
    for (int i = 0; i < i_points; i++) {
        for (int j = 0; j < r_points; j++) {
            double real_part = min + dR * j;
            double imag_part = max - dI * i;
            h_input[i_points * i + j] = real_part + imag_part * I;
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
            z_temp = 0.0 + 0.0 * I;
            result = 1;
            for (int k = 0; k < MAX_ITER; k++) {
                z_temp = z_temp * z_temp + c;
                if (cabs(z_temp) > 2.0) {
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

    // number of points outside
    printf("The number of points outside is: %d\n", num_outside);

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