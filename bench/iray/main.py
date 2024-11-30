import numpy as np
import time
import psutil
import os

# Bellek kullanımını ölçen fonksiyon
def memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024  # KB cinsinden

# Zaman ölçümünü başlat ve durdur
def start_timer():
    return time.time()

def end_timer(start):
    print(f"Time: {time.time() - start:.6f} seconds")

# Benchmark fonksiyonları
def benchmark_add():
    print("\n--- Benchmark: Add ---")
    A = np.ones((1000, 1000))
    B = np.ones((1000, 1000)) * 2

    print(f"Memory usage before addition: {memory_usage()} KB")
    start = start_timer()
    C = A + B
    end_timer(start)
    print(f"Memory usage after addition: {memory_usage()} KB")

def benchmark_dot():
    print("\n--- Benchmark: Dot ---")
    A = np.ones((1000, 1000))
    B = np.ones((1000, 1000)) * 2

    print(f"Memory usage before dot: {memory_usage()} KB")
    start = start_timer()
    C = np.dot(A, B)
    end_timer(start)
    print(f"Memory usage after dot: {memory_usage()} KB")

def apply(x):
    return x ** 2

def benchmark_apply():
    print("\n--- Benchmark: Apply ---")
    A = np.ones((1000, 1000))

    print(f"Memory usage before apply: {memory_usage()} KB")
    start = start_timer()
    B = np.vectorize(apply)(A)
    end_timer(start)
    print(f"Memory usage after apply: {memory_usage()} KB")

def benchmark_scale():
    print("\n--- Benchmark: Scale ---")
    A = np.ones((1000, 1000))

    print(f"Memory usage before scale: {memory_usage()} KB")
    start = start_timer()
    B = A * 0.5
    end_timer(start)
    print(f"Memory usage after scale: {memory_usage()} KB")

def benchmark_max():
    print("\n--- Benchmark: Max ---")
    A = np.ones((1000, 1000))

    print(f"Memory usage before max: {memory_usage()} KB")
    start = start_timer()
    max_value = np.max(A)
    end_timer(start)
    print(f"Memory usage after max: {memory_usage()} KB")

def benchmark_min():
    print("\n--- Benchmark: Min ---")
    A = np.ones((1000, 1000))

    print(f"Memory usage before min: {memory_usage()} KB")
    start = start_timer()
    min_value = np.min(A)
    end_timer(start)
    print(f"Memory usage after min: {memory_usage()} KB")

def benchmark_transpose():
    print("\n--- Benchmark: Transpose ---")
    A = np.ones((1000, 1000))

    print(f"Memory usage before transpose: {memory_usage()} KB")
    start = start_timer()
    B = A.T
    end_timer(start)
    print(f"Memory usage after transpose: {memory_usage()} KB")

# Ana fonksiyon
if __name__ == "__main__":
    benchmark_add()
    benchmark_dot()
    benchmark_apply()
    benchmark_scale()
    benchmark_max()
    benchmark_min()
    benchmark_transpose()
