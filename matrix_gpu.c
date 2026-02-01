#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>


__global__ void matrixMultiplyGPU( float *A, float *B, float *C, int N){
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (row < N && col < N){
		float sum = 0.0f;
		for (int k=0; k<N; k++){
			sum += A[row*N + k] * B[k*N + col];
		}
		C[row*N + col] = sum;
	}
}


int main(int argc, char **argv){
        int N = (argc > 1) ? atoi(argv[1]) : 1024;
        size_t size = N * N * (sizeof(float));

        float *host_A = (float *)malloc(size);
        float *host_B = (float *)malloc(size);
        float *host_C = (float *)malloc(size);

        for(int i=0; i<N*N; i++){
                host_A[i] = rand() % 100/100.0f;
                host_B[i] = rand() % 100/100.0f;
        }
	float *virt_A, *virt_B, *virt_C;
	cudaMalloc((void**)&virt_A, size);
	cudaMalloc((void**)&virt_B, size);
	cudaMalloc((void**)&virt_C, size);

	cudaMemcpy(virt_A, host_A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(virt_B, host_B, size, cudaMemcpyHostToDevice);

	dim3 threadsPerBlock(16,16);
	dim3 blocksPerGrid((N+15)/16, (N+15)/16);

        clock_t start = clock();
        matrixMultiplyGPU<<<blocksPerGrid, threadsPerBlock>>>(virt_A, virt_B, virt_C, N);
	
	cudaDeviceSynchronize();
        clock_t end = clock();

	cudaMemcpy(host_C, virt_C, size, cudaMemcpyDeviceToHost);

        double elapsed = (double) (end - start) / CLOCKS_PER_SEC;
        printf("GPU execution time (N=%d): %f seconds\n", N, elapsed);

	cudaFree(virt_A); cudaFree(virt_B); cudaFree(virt_C);
        free(host_A); free(host_B); free(host_C);
        return 0;
}
