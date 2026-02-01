import ctypes
import numpy as np
import time
import argparse
from PIL import Image

# Load the shared library
lib = ctypes.CDLL('./libmatrix.so')

# Define the function signatures
lib.matrix_multiply.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # h_A
    ctypes.POINTER(ctypes.c_float),  # h_B
    ctypes.POINTER(ctypes.c_float),  # h_C
    ctypes.c_int  # N
]

lib.convolute_image.argtypes = [
    ctypes.POINTER(ctypes.c_ubyte),  # h_input
    ctypes.POINTER(ctypes.c_ubyte),  # h_output
    ctypes.c_int,  # width
    ctypes.c_int,  # height
    ctypes.c_int,  # N
    ctypes.POINTER(ctypes.c_float)  # h_kernel
]

def matrix_multiply_python(A, B, C, N):
    A_ptr = A.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    B_ptr = B.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    C_ptr = C.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    lib.matrix_multiply(A_ptr, B_ptr, C_ptr, N)

def convolute_image_python(input_img, output_img, width, height, N, kernel):
    input_ptr = input_img.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
    output_ptr = output_img.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
    kernel_ptr = kernel.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    lib.convolute_image(input_ptr, output_ptr, width, height, N, kernel_ptr)

def get_kernel(N, filter_type):
    if filter_type == 'blur':
        return np.full((N, N), 1.0 / (N * N), dtype=np.float32).flatten()
    elif filter_type == 'edge':
        # Simple edge kernel (approximation)
        kernel = np.zeros((N, N), dtype=np.float32)
        center = N // 2
        kernel[center, center] = N * N - 1
        kernel -= 1.0 / (N * N)
        return kernel.flatten()
    else:
        raise ValueError("Unsupported filter type")

def main():
    parser = argparse.ArgumentParser(description="Test CUDA matrix multiplication and image convolution")
    parser.add_argument('--task', choices=['matrix', 'convolution'], required=True, help='Task to perform')
    
    # Matrix options
    parser.add_argument('--sizes', nargs='+', type=int, default=[512, 1024], help='Matrix sizes for matrix task')
    
    # Convolution options
    parser.add_argument('--input', help='Input image file for convolution')
    parser.add_argument('--output', help='Output image file for convolution')
    parser.add_argument('--N', type=int, default=3, help='Kernel size for convolution')
    parser.add_argument('--filter', choices=['blur', 'edge'], default='blur', help='Filter type for convolution')
    
    args = parser.parse_args()
    
    if args.task == 'matrix':
        print("Testing Matrix Multiplication:")
        for N in args.sizes:
            print(f"  N={N}")
            
            A = np.random.rand(N, N).astype(np.float32)
            B = np.random.rand(N, N).astype(np.float32)
            C = np.zeros((N, N), dtype=np.float32)
            
            start = time.time()
            matrix_multiply_python(A, B, C, N)
            end = time.time()
            
            elapsed = end - start
            print(f"    Time: {elapsed:.4f} seconds")
    
    elif args.task == 'convolution':
        if not args.input or not args.output:
            print("Error: --input and --output required for convolution")
            return
        
        print("Testing Image Convolution:")
        print(f"  Input: {args.input}")
        print(f"  Output: {args.output}")
        print(f"  N: {args.N}, Filter: {args.filter}")
        
        # Load image
        img = Image.open(args.input).convert('L')  # Grayscale
        width, height = img.size
        input_img = np.array(img, dtype=np.uint8)
        output_img = np.zeros_like(input_img)
        
        kernel = get_kernel(args.N, args.filter)
        
        start = time.time()
        convolute_image_python(input_img, output_img, width, height, args.N, kernel)
        end = time.time()
        
        elapsed = end - start
        print(f"  Convolution time: {elapsed:.4f} seconds")
        
        # Save output
        out_img = Image.fromarray(output_img)
        out_img.save(args.output)
        print(f"  Output saved to {args.output}")

if __name__ == '__main__':
    main()