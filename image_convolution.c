#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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

/* -------------------------
   Optimized convolution (no bounds checks in inner loop):
   - Writes 0 on borders
   - Computes only interior pixels
   ------------------------- */
void convolute(const unsigned char *input,
               unsigned char *output,
               int width,
               int height,
               int N,
               const float *kernel)
{
    int r = N / 2;

    // For simplicity we handle full image; borders set to 0
    // Initialize output to 0
    memset(output, 0, (size_t)width * (size_t)height);

    // Compute only pixels where kernel fits
    for (int y = r; y < height - r; y++) {
        for (int x = r; x < width - r; x++) {

            float sum = 0.0f;
            const float *kptr = kernel;

            // Pointer to top-left of the kernel window in input
            const unsigned char *base = input + (y - r) * width + (x - r);

            // ky/kx loops
            for (int ky = 0; ky < N; ky++) {
                const unsigned char *row = base + ky * width;
                for (int kx = 0; kx < N; kx++) {
                    sum += (float)row[kx] * (*kptr++);
                }
            }

            // clamp
            if (sum < 0.0f) sum = 0.0f;
            if (sum > 255.0f) sum = 255.0f;

            output[y * width + x] = (unsigned char)sum;
        }
    }
}

int main(int argc, char **argv) {
    // Required:
    //   N filter image_path output_path
    //
    // Example:
    //   ./conv_cpu 3 edge input.pgm output.pgm

    if (argc < 5) {
        printf("Usage: %s <N=3|5|7> <filter=edge|blur|blue> <input.pgm> <output.pgm>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    const char *filter = argv[2];
    const char *in_path = argv[3];
    const char *out_path = argv[4];

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

    unsigned char *output = (unsigned char*)malloc((size_t)w * (size_t)h);
    if (!output) {
        free(input);
        printf("Error: malloc output failed\n");
        return 1;
    }

    clock_t start = clock();
    convolute(input, output, w, h, N, kernel);
    clock_t end = clock();

    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("CPU convolution time: %f seconds | image=%dx%d | N=%d | filter=%s\n",
           elapsed, w, h, N, filter);

    if (!write_pgm_p5(out_path, output, w, h)) {
        printf("Error: Failed to write output image: %s\n", out_path);
        free(input);
        free(output);
        return 1;
    }

    free(input);
    free(output);
    return 0;
}
