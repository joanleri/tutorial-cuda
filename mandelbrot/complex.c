/* 
*
*   Ejemplos de números complejos con C
*
*/

#include <stdio.h>
#include <complex.h>

int main() {
    double complex z1 = 5.0 + 12.0 * I;
    double complex z2 = 1.0 - 4.0 * I;

    // Adding
    double complex sum = z1 + z2;

    // Substracting
    double complex difference = z1 - z2;

    // Multiplication
    double complex product = z1 * z2;

    // Division
    double complex quotient = z1 / z2;

    // Conjugate
    double complex conjugate = conj(z1);

    // Distance from the origin
    float distance = cabs(z1);
    printf("Distance from zero is: %f\n", distance);

    return 0;
}