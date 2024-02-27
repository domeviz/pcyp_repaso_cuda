#include <iostream>
#include <cuda.h>

#define VECTOR_ELEMENTS 2048

extern "C"
{
__declspec(dllexport) unsigned long NvOptimusEnablement = 0x00000001;
}

__global__
void vecAdd(float *d_A, float *d_B, float *d_C, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < n)
        d_C[index] = d_A[index] + d_B[index];
}

int main() {

    // host
    float *h_A = new float[VECTOR_ELEMENTS];
    float *h_B = new float[VECTOR_ELEMENTS];
    float *h_C = new float[VECTOR_ELEMENTS];

    // Device
    float *d_A, *d_B, *d_C;
    int size = VECTOR_ELEMENTS * sizeof(float);
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Inicializar vectores
    for (int i = 0; i < VECTOR_ELEMENTS; i++) {
        h_A[i] = 1.f;
        h_B[i] = 2.f;
        h_C[i] = 0.f;
    }

    // Linea para probar el consumo de memoria de la GPU
    //cudaMalloc(&xx, 1024 * 1024 * 1024 * sizeof(float));

    // Copiar host-to-device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // invocar al kernel
    vecAdd<<<8, 256>>>(d_A, d_B, d_C, VECTOR_ELEMENTS);

    // copiar devide-to-host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 15; i++) {
        printf("%.0f, ", h_C[i]);
    }

    return 0;
}