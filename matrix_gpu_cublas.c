#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>



int main(int argc, char **argv) {
  int N = (argc > 1) ? atoi(argv[1]) : 1024;
  size_t size = N*N*sizeof(float);

  //allocate CPU memory
  float *A = (float *)malloc(size);
  float *B = (float *)malloc(size);
  float *C = (float *)malloc(size);

  //Initialize CPU matrices with random values
  for (int i = 0; i < N*N; i++) {
    A[i] = rand() % 100 / 100.0f;
    B[i] = rand() % 100 / 100.0f;
  }

  //allocate GPU memory
  float *d_A, *d_B, *d_C;
  cudaMalloc((void**)&d_A, size);
  cudaMalloc((void**)&d_B, size);
  cudaMalloc((void**)&d_C, size);

  //copy data from CPU to GPU
  cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);


  cublasHandle_t handle;
  cublasCreate(&handle);

  float alpha = 1.0f;
  float beta = 0.0f;

  //warm up run
  cublasSgemm(handle,
    CUBLAS_OP_N, CUBLAS_OP_N,
    N, N, N,
    &alpha,
    d_B, N,
    d_A, N,
    &beta,
    d_C, N);
  cudaDeviceSynchronize();


  //timed run
  cudaEvent_t start, stop;
  cudaEventCreate(&start);//create CUDA event to makr the start time
  cudaEventCreate(&stop);

  cudaEventRecord(start); //record the start event

  cublasSgemm(handle,
    CUBLAS_OP_N, CUBLAS_OP_N,
    N, N, N,
    &alpha,
    d_B, N,
    d_A, N,
    &beta,
    d_C, N);

  cudaEventRecord(stop); //record the stop event
  cudaEventSynchronize(stop);


  float milliseconds = 0; //variable to store the time
  cudaEventElapsedTime(&milliseconds, start, stop); //compute time between start and stop, result goes into milliseconds

  //copy data from GPU to CPU
  cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

  //copy timing result
  printf("cuBLAS execution time (N=%d): %f seconds\n", N, milliseconds/1000.0f);


  //cleanup
  cublasDestroy(handle);
  cudaFree(d_A);//gpu part
  cudaFree(d_B);
  cudaFree(d_C);
  free(A);//cpu part
  free(B);
  free(C);

  return 0;
}
