#include <stdio.h>
#include <stdlib.h>
const int ARR_SIZE = 64;

// Función para calcular el cuadrado de los elementos
void cuadrado(float *d_out, float *d_in) {
	int index = 0;
	for(index = 0; index < ARR_SIZE; index++)
		d_out[index] = d_in[index]*d_in[index];
}

int main(int argc, char **argv){
	
	// Espacio para arreglos original y resultado
	float *h_orig, *h_res;

	
	//Reserva espacio para arreglos locales
	h_orig = (float  *)malloc(ARR_SIZE*sizeof(float));
	h_res  = (float  *)malloc(ARR_SIZE*sizeof(float));


	for(int i = 0; i < ARR_SIZE; i++){
	  h_orig[i]= (float)i;
	}

	// Calcula cuadrados invocando a la función
	cuadrado(h_res, h_orig); 

	//Despliega resultado

	for(int i=0;i<ARR_SIZE; i++){
	  printf("%04.2f",h_res[i]);
	  printf("%c",((i%5)<4) ? '\t':'\n');
	}
	 
	//libera memoria y termina
	free(h_orig);
	free(h_res);

	return(0);
}
