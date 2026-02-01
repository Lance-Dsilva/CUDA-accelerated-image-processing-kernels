
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

__global__ void matrixMultiplyGPU( float *A, float *B, float *C, int N){
        __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
	__shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

	int bx = blockIdx.x; int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;

	int Row = by * TILE_WIDTH + ty;
	int Col = bx * TILE_WIDTH + tx;
	
	float Pvalue = 0.0;
	for (int m=0; m< (N + TILE_WIDTH -1) / TILE_WIDTH; ++m){
		if (Row < N && (m*TILE_WIDTH+tx) < N)
			ds_A[ty][tx] = A[Row * N + m * TILE_WIDTH + tx];
		else	
			ds_A[ty][tx] = 0.0f;
		if (Col < N && (m*TILE_WIDTH+ty) < N)
			ds_B[ty][tx] = B[(m*TILE_WIDTH + ty) * N + Col];
		else
			ds_B[ty][tx] = 0.0f;

		__syncthreads();
		
		for (int k=0; k< TILE_WIDTH; ++k)
			Pvalue += ds_A[ty][k] * ds_B[k][tx];
		__syncthreads();
	}
	if (Row < N && Col < N)
		C[Row * N + Col] = Pvalue;
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

        dim3 threadsPerBlock(TILE_WIDTH,TILE_WIDTH);
        dim3 blocksPerGrid((N+TILE_WIDTH-1)/TILE_WIDTH, (N+TILE_WIDTH-1)/TILE_WIDTH);

        clock_t start = clock();
        matrixMultiplyGPU<<<blocksPerGrid, threadsPerBlock>>>(virt_A,virt_B,virt_C,N);

        cudaDeviceSynchronize();
        clock_t end = clock();

        cudaMemcpy(host_C, virt_C, size, cudaMemcpyDeviceToHost);

        double elapsed = (double) (end - start) / CLOCKS_PER_SEC;
        printf("GPU execution time (N=%d): %f seconds\n", N, elapsed);

        cudaFree(virt_A); cudaFree(virt_B); cudaFree(virt_C);
        free(host_A); free(host_B); free(host_C);
        return 0;
}
