#include <stdio.h>
#include <stdlib.h>
const int ARR_SIZE = 64;
const int ARR_BYTES = ARR_SIZE*sizeof(float);

__global__ 
void cuadrado(float* d_out, float* d_in) {
	int idx = threadIdx.x;
	float f = d_in[idx];
	dout[idx] = f*f;
}

int main(int argc, char **argv){
	
	// Apuntadores a arreglos en host y en device
	float *h_orig, *h_res;
	float *d_in, *d_out;
	
	//Reserva espacio para arreglos 
	h_orig = (float  *)malloc(ARR_SIZE*sizeof(float));
	h_res  = (float  *)malloc(ARR_SIZE*sizeof(float));
	cudaMalloc((void**) &d_in, ARR_BYTES);
	cudaMalloc((void**) &d_out, ARR_BYTES);
	
	for(int i=0;i< ARR_SIZE; i++){  // Llena arreglo inicial
		h_orig[i]= (float)i;
	}

	//Transfiere arreglo a device
	cudaMemcpy(d_in, h_orig, ARR_BYTES, cudaMemcpyHostToDevice);
	
	//Lanza el kernel
	cuadrado<<<1,ARR_SZ>>>(d_out,d_in);
	
	//Toma el resultado
	cudaMemcpy(h_res, d_out, ARR_BYTES, cudaMemcpyDeviceToHost);

	//Despliega resultado
	for(int i=0;i<ARR_SIZE; i++){
	  printf("%4.2f",h_res[i]);
	  printf("%c",((i%5)<4) ? '\t':'\n');
	}
	 
	//libera memoria y termina
	free(h_orig);
	free(h_res);
	cudaFree(d_in);
	cudaFree(d_out);

	return(0);
}
