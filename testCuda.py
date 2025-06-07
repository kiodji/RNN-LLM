import numpy as np
from numba import cuda
from timeit import default_timer as timer   

n = 16384 # matrix side size
threads_per_block = 256
blocks = int(n / threads_per_block)

# Input Matrix
a = np.ones(n*n).reshape(n, n).astype(np.float32)
# Here we set an arbitrary row to an arbitrary value to facilitate a check for correctness below.
a[3] = 9

# Output vector
sums = np.zeros(n).astype(np.float32)

d_a = cuda.to_device(a)
d_sums = cuda.to_device(sums)


@cuda.jit
def row_sums(a, sums, n):
    idx = cuda.grid(1)
    sum = 0.0
    
    for i in range(n):
        # Each thread will sum a row of `a`
        sum += a[idx][i]
        
    sums[idx] = sum
 
start = timer() 
row_sums[blocks, threads_per_block](d_a, d_sums, n); cuda.synchronize()
print("with GPU:", timer()-start) 