import os
import numpy as np
from numba import int32, cuda, njit
import numba
import timeit


def hist_cpu(A):
    """
     Returns
     -------
     np.array
         histogram of A of size 256
     """
    C = np.zeros(shape=256, dtype=np.int32)
    shape = A.shape[0]
    for i in range(shape):
        C[A[i]] += 1
    return C

@njit
def hist_numba(A):
    """
     Returns
     -------
     np.array
         histogram of A of size 256
     """
    shape = A.shape[0]
    C = np.zeros(shape=256, dtype=np.int32)
    for i in numba.prange(shape):
        C[A[i]] += 1
   
    return C

def hist_gpu(A):
    # Allocate the output np.array histogram C in GPU memory using cuda.to_device
    #
    # invoke the hist kernel with 1000 threadBlocks with 1024 threads each
    #
    # copy the output histogram C from GPU to cpu using copy_to_host()
    
    C = np.zeros(shape=256, dtype=np.uint64)
    d_A = cuda.to_device(A)
    d_C = cuda.to_device(C)

    blocks_per_grid = 1000
    threads_per_block = 1024
    hist_kernel[blocks_per_grid, threads_per_block](d_A, d_C)
    
    C = d_C.copy_to_host()
    return C

@cuda.jit
def hist_kernel(A, C):
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bw = cuda.blockDim.x
    pos = tx + bx*bw

    if pos < A.size and A[pos] < C.shape[0]:
        cuda.atomic.add(C, A[pos], 1)
    

#this is the comparison function - keep it as it is, don't change A.
def hist_comparison():
    A = np.random.randint(0,256,1000*1024)

    def timer(f):
        return min(timeit.Timer(lambda: f(A)).repeat(3, 20))

    print('CPU:', timer(hist_cpu))
    print('Numba:', timer(hist_numba))
    print('CUDA:', timer(hist_gpu))

if __name__ == '__main__':
    os.environ['NUMBAPRO_NVVM'] = '/usr/local/cuda-9.0/nvvm/lib64/libnvvm.so'
    os.environ['NUMBAPRO_LIBDEVICE'] = '/usr/local/cuda-9.0/nvvm/libdevice/'

    hist_comparison()
