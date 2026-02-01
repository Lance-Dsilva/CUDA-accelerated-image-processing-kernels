import subprocess
import os
import sys
import ctypes
import numpy as np
from PIL import Image

# For python mode
lib = ctypes.CDLL('./libmatrix.so')
lib.convolute_image.argtypes = [
    ctypes.POINTER(ctypes.c_ubyte),
    ctypes.POINTER(ctypes.c_ubyte),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_float)
]

def convolute_image_python(input_img, output_img, width, height, N, kernel):
    input_ptr = input_img.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
    output_ptr = output_img.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
    kernel_ptr = kernel.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    lib.convolute_image(input_ptr, output_ptr, width, height, N, kernel_ptr)

def get_kernel(N, filter_type):
    if filter_type == 'blur':
        return np.full((N, N), 1.0 / (N * N), dtype=np.float32).flatten()
    elif filter_type == 'edge':
        kernel = np.zeros((N, N), dtype=np.float32)
        center = N // 2
        kernel[center, center] = N * N - 1
        kernel -= 1.0 / (N * N)
        return kernel.flatten()
    else:
        raise ValueError("Unsupported filter type")

def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ['cpu', 'gpu', 'python']:
        print("Usage: python image_caller.py cpu|gpu|python")
        sys.exit(1)
    
    mode = sys.argv[1]
    folder = mode
    os.makedirs(folder, exist_ok=True)
    
    images = ["clock_256.pgm","boat_512.pgm", "male_1024.pgm"]
    filters = ["blur", "edge"]
    ns = [3, 5, 7]
    
    if mode in ['cpu', 'gpu']:
        # Compile
        if mode == 'cpu':
            print("Compiling CPU version...")
            subprocess.run(["gcc", "image_convolution.c", "-o", "image_conv"], check=True)
            exe = "./image_conv"
        else:
            print("Compiling GPU version...")
            subprocess.run(["nvcc", "-x", "cu", "image_convolution_gpu.c", "-o", "image_conv_gpu"], check=True)
            exe = "./image_conv_gpu"
        
        for input_image in images:
            print(f"Processing {input_image} with {mode}")
            for filter_type in filters:
                for n in ns:
                    output_image = f"{folder}/{os.path.splitext(input_image)[0]}_output_{filter_type}_{n}_{mode}.pgm"
                    print(f"  Running with filter={filter_type}, N={n}")
                    subprocess.run([exe, str(n), filter_type, input_image, output_image], check=True)
    
    elif mode == 'python':
        for input_image in images:
            print(f"Processing {input_image} with {mode}")
            img = Image.open(input_image).convert('L')
            width, height = img.size
            input_img = np.array(img, dtype=np.uint8)
            
            for filter_type in filters:
                for n in ns:
                    output_img = np.zeros_like(input_img)
                    kernel = get_kernel(n, filter_type)
                    
                    print(f"  Running with filter={filter_type}, N={n}")
                    convolute_image_python(input_img, output_img, width, height, n, kernel)
                    
                    output_filename = f"{folder}/{os.path.splitext(input_image)[0]}_output_{filter_type}_{n}_{mode}.png"
                    out_img = Image.fromarray(output_img)
                    out_img.save(output_filename)
                    print(f"    Saved to {output_filename}")
    
    print(f"All {mode} convolutions completed. Outputs in {folder}/")

if __name__ == '__main__':
    main()
