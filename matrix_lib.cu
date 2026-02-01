#include <cuda_runtime.h>

#define TILE_WIDTH 16

__global__ void matrixMultiplyGPU(float *A, float *B, float *C, int N) {
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Pvalue = 0.0;
    for (int m = 0; m < (N + TILE_WIDTH - 1) / TILE_WIDTH; ++m) {
        if (Row < N && (m * TILE_WIDTH + tx) < N)
            ds_A[ty][tx] = A[Row * N + m * TILE_WIDTH + tx];
        else
            ds_A[ty][tx] = 0.0f;
        if (Col < N && (m * TILE_WIDTH + ty) < N)
            ds_B[ty][tx] = B[(m * TILE_WIDTH + ty) * N + Col];
        else
            ds_B[ty][tx] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k)
            Pvalue += ds_A[ty][k] * ds_B[k][tx];
        __syncthreads();
    }
    if (Row < N && Col < N)
        C[Row * N + Col] = Pvalue;
}

__global__ void convolute_kernel(const unsigned char *input, unsigned char *output, int width, int height, int N, const float *kernel) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int r = N / 2;
    if (x < r || x >= width - r || y < r || y >= height - r) {
        output[y * width + x] = 0;
        return;
    }

    float sum = 0.0f;
    for (int ky = 0; ky < N; ky++) {
        for (int kx = 0; kx < N; kx++) {
            int ix = x - r + kx;
            int iy = y - r + ky;
            sum += (float)input[iy * width + ix] * kernel[ky * N + kx];
        }
    }

    if (sum < 0.0f) sum = 0.0f;
    if (sum > 255.0f) sum = 255.0f;
    output[y * width + x] = (unsigned char)sum;
}

extern "C" void matrix_multiply(float *h_A, float *h_B, float *h_C, int N) {
    size_t size = N * N * sizeof(float);

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 blocksPerGrid((N + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);

    matrixMultiplyGPU<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

extern "C" void convolute_image(unsigned char *h_input, unsigned char *h_output, int width, int height, int N, float *h_kernel) {
    size_t img_size = (size_t)width * height;
    size_t kernel_size = N * N * sizeof(float);

    unsigned char *d_input, *d_output;
    float *d_kernel;

    cudaMalloc((void**)&d_input, img_size);
    cudaMalloc((void**)&d_output, img_size);
    cudaMalloc((void**)&d_kernel, kernel_size);

    cudaMemcpy(d_input, h_input, img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernel_size, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    convolute_kernel<<<gridDim, blockDim>>>(d_input, d_output, width, height, N, d_kernel);

    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, img_size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
}