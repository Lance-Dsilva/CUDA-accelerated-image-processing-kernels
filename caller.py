import subprocess
import os

#Compile the C programs
compile_cpu_cmd = ["gcc", "matrix_cpu.c", "-o", "matrix_cpu", "-O2"]
subprocess.run(compile_cpu_cmd, check=True)

compile_gpu_cmd = ["nvcc", "-x", "cu", "matrix_gpu.c", "-o", "matrix_gpu"]
subprocess.run(compile_gpu_cmd, check=True)

compile_gpu_opt_cmd = ["nvcc", "-x", "cu", "matrix_gpu_optimized.c", "-o", "matrix_gpu_optimized"]
subprocess.run(compile_gpu_opt_cmd, check=True)

compile_gpu_cublas_cmd = ["nvcc", "-x", "cu", "matrix_gpu_cublas.c", "-o", "matrix_gpu_cublas","-lcublas"]
subprocess.run(compile_gpu_cublas_cmd, check=True)

# List of N values
n_values = ["256","512", "1024", "2048"]

# Run each program with the same N values
for n in n_values:
    print(f"Running matrix_cpu with N={n}")
    run_cmd = ["./matrix_cpu", n]
    subprocess.run(run_cmd, check=True)

    print(f"Running matrix_gpu with N={n}")
    run_cmd = ["./matrix_gpu", n]
    subprocess.run(run_cmd, check=True)

    print(f"Running matrix_gpu_optimized with N={n}")
    run_cmd = ["./matrix_gpu_optimized", n]
    subprocess.run(run_cmd, check=True)

    print(f"Running matrix_gpu_cublas with N={n}")
    run_cmd = ["./matrix_gpu_cublas", n]
    subprocess.run(run_cmd, check=True)