#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>

/* -------------------------
   Global kernels (edge + blur) for N = 3, 5, 7
   (We keep them global as you requested.)
   ------------------------- */

static const float EDGE_3[3][3] = {
    {-1, -1, -1},
    {-1,  8, -1},
    {-1, -1, -1}
};
static const float BLUR_3[3][3] = {
    {1.f/9, 1.f/9, 1.f/9},
    {1.f/9, 1.f/9, 1.f/9},
    {1.f/9, 1.f/9, 1.f/9}
};

static const float EDGE_5[5][5] = {
    { 0,  0, -1,  0,  0},
    { 0, -1, -2, -1,  0},
    {-1, -2, 16, -2, -1},
    { 0, -1, -2, -1,  0},
    { 0,  0, -1,  0,  0}
};
static const float BLUR_5[5][5] = {
    {1.f/25, 1.f/25, 1.f/25, 1.f/25, 1.f/25},
    {1.f/25, 1.f/25, 1.f/25, 1.f/25, 1.f/25},
    {1.f/25, 1.f/25, 1.f/25, 1.f/25, 1.f/25},
    {1.f/25, 1.f/25, 1.f/25, 1.f/25, 1.f/25},
    {1.f/25, 1.f/25, 1.f/25, 1.f/25, 1.f/25}
};

static const float EDGE_7[7][7] = {
    { 0,  0, -1, -1, -1,  0,  0},
    { 0, -1, -2, -2, -2, -1,  0},
    {-1, -2, -1,  0, -1, -2, -1},
    {-1, -2,  0, 20,  0, -2, -1},
    {-1, -2, -1,  0, -1, -2, -1},
    { 0, -1, -2, -2, -2, -1,  0},
    { 0,  0, -1, -1, -1,  0,  0}
};
static const float BLUR_7[7][7] = {
    {1.f/49,1.f/49,1.f/49,1.f/49,1.f/49,1.f/49,1.f/49},
    {1.f/49,1.f/49,1.f/49,1.f/49,1.f/49,1.f/49,1.f/49},
    {1.f/49,1.f/49,1.f/49,1.f/49,1.f/49,1.f/49,1.f/49},
    {1.f/49,1.f/49,1.f/49,1.f/49,1.f/49,1.f/49,1.f/49},
    {1.f/49,1.f/49,1.f/49,1.f/49,1.f/49,1.f/49,1.f/49},
    {1.f/49,1.f/49,1.f/49,1.f/49,1.f/49,1.f/49,1.f/49},
    {1.f/49,1.f/49,1.f/49,1.f/49,1.f/49,1.f/49,1.f/49}
};

static const float* pick_kernel(const char *filter, int N) {
    int is_edge = (strcmp(filter, "edge") == 0);
    int is_blur = (strcmp(filter, "blur") == 0 || strcmp(filter, "blue") == 0);

    if (!is_edge && !is_blur) return NULL;

    if (N == 3) return is_edge ? &EDGE_3[0][0] : &BLUR_3[0][0];
    if (N == 5) return is_edge ? &EDGE_5[0][0] : &BLUR_5[0][0];
    if (N == 7) return is_edge ? &EDGE_7[0][0] : &BLUR_7[0][0];
    return NULL;
}

/* -------------------------
   CUDA kernel for convolution
   ------------------------- */
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

/* -------------------------
   Host function to launch kernel
   ------------------------- */
void convolute(unsigned char *d_input, unsigned char *d_output, int width, int height, int N, float *d_kernel) {
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
    convolute_kernel<<<gridDim, blockDim>>>(d_input, d_output, width, height, N, d_kernel);
    cudaDeviceSynchronize();
}

/* -------------------------
   Read PGM (P5) grayscale image.
   Returns allocated pixel buffer, sets width/height.
   ------------------------- */
static unsigned char* read_pgm_p5(const char *path, int *w, int *h) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;

    char magic[3] = {0};
    if (fscanf(fp, "%2s", magic) != 1) { fclose(fp); return NULL; }
    if (strcmp(magic, "P5") != 0) { fclose(fp); return NULL; }

    // Skip comment lines
    int c = fgetc(fp);
    while (c == '#') {
        while (c != '\n' && c != EOF) c = fgetc(fp);
        c = fgetc(fp);
    }
    ungetc(c, fp);

    int width = 0, height = 0, maxval = 0;
    if (fscanf(fp, "%d %d", &width, &height) != 2) { fclose(fp); return NULL; }
    if (fscanf(fp, "%d", &maxval) != 1) { fclose(fp); return NULL; }
    if (maxval != 255) { fclose(fp); return NULL; }

    fgetc(fp); // consume single whitespace after header

    size_t bytes = (size_t)width * (size_t)height;
    unsigned char *data = (unsigned char*)malloc(bytes);
    if (!data) { fclose(fp); return NULL; }

    size_t got = fread(data, 1, bytes, fp);
    fclose(fp);
    if (got != bytes) { free(data); return NULL; }

    *w = width; *h = height;
    return data;
}

static int write_pgm_p5(const char *path, const unsigned char *data, int w, int h) {
    FILE *fp = fopen(path, "wb");
    if (!fp) return 0;
    fprintf(fp, "P5\n%d %d\n255\n", w, h);
    fwrite(data, 1, (size_t)w * (size_t)h, fp);
    fclose(fp);
    return 1;
}

int main(int argc, char **argv) {
    // Required:
    //   N filter image_path output_path [iterations]
    //
    // Example:
    //   ./conv_gpu 3 blur input.pgm output.pgm 1

    if (argc < 5) {
        printf("Usage: %s <N=3|5|7> <filter=edge|blur|blue> <input.pgm> <output.pgm> [iterations=1]\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    const char *filter = argv[2];
    const char *in_path = argv[3];
    const char *out_path = argv[4];
    int iterations = (argc > 5) ? atoi(argv[5]) : 1;

    const float *kernel = pick_kernel(filter, N);
    if (!kernel) {
        printf("Error: Supported N = 3,5,7 and filter = edge|blur|blue\n");
        return 1;
    }

    int w = 0, h = 0;
    unsigned char *input = read_pgm_p5(in_path, &w, &h);
    if (!input) {
        printf("Error: Failed to read input image. Must be binary grayscale PGM (P5) maxval 255.\n");
        return 1;
    }

    size_t size = (size_t)w * (size_t)h;
    unsigned char *output = (unsigned char*)malloc(size);
    if (!output) {
        free(input);
        printf("Error: malloc output failed\n");
        return 1;
    }

    // GPU allocations
    unsigned char *d_buf1, *d_buf2;
    float *d_kernel;
    cudaMalloc((void**)&d_buf1, size);
    cudaMalloc((void**)&d_buf2, size);
    cudaMalloc((void**)&d_kernel, N * N * sizeof(float));

    cudaMemcpy(d_buf1, input, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, N * N * sizeof(float), cudaMemcpyHostToDevice);
    free(input);

    unsigned char *d_current = d_buf1;
    unsigned char *d_next = d_buf2;

    clock_t start = clock();
    for (int i = 0; i < iterations; i++) {
        convolute(d_current, d_next, w, h, N, d_kernel);
        // Swap buffers
        unsigned char *temp = d_current;
        d_current = d_next;
        d_next = temp;
    }
    clock_t end = clock();

    cudaMemcpy(output, d_current, size, cudaMemcpyDeviceToHost);

    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("GPU convolution time: %f seconds | image=%dx%d | N=%d | filter=%s | iterations=%d\n",
           elapsed, w, h, N, filter, iterations);

    if (!write_pgm_p5(out_path, output, w, h)) {
        printf("Error: Failed to write output image: %s\n", out_path);
        free(output);
        cudaFree(d_buf1);
        cudaFree(d_buf2);
        cudaFree(d_kernel);
        return 1;
    }

    free(output);
    cudaFree(d_buf1);
    cudaFree(d_buf2);
    cudaFree(d_kernel);
    return 0;
}
