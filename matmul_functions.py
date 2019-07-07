import os
import numpy as np
from numba import njit, cuda
import numba
import timeit
import math


def matmul_trivial(X, Y):
    
    Z = np.zeros(shape=(X.shape[0], Y.shape[1]),
                 dtype=np.float32)
    for i in range(X.shape[0]):
        for k in range(Y.shape[1]):
            for j in range(X.shape[1]):
                Z[i][k] += X[i][j]*Y[j][k]

    return Z



@njit(parallel=True, fastmath=True)
def matmul_numba(X, Y):
    Z = np.zeros(shape=(X.shape[0], Y.shape[1]),
                 dtype=np.float32)
    
    # Only the outermost loop is parallelized. Both inner loops
    # are treated as a single range function (from Docs).
    for i in numba.prange(X.shape[0]):
        for k in numba.prange(Y.shape[1]):
            for j in numba.prange(X.shape[1]):
                Z[i][k] += X[i][j]*Y[j][k]
    return Z


def matmul_gpu(X, Y):
    # Allocate the output matrix in GPU memory using cuda.to_device
    #
    # invoke the dot kernel with 1 threadBlock with 1024 threads
    #
    # copy the output matrix from GPU to cpu using copy_to_host()
    d_X = cuda.to_device(X)
    d_Y = cuda.to_device(Y)
    Z = np.zeros(shape=(X.shape[0], Y.shape[1]),
                 dtype=np.float32)
    d_Z = cuda.to_device(Z)
    threads_per_block = 1024
    blocks_per_grid = 1

    matmul_kernel[blocks_per_grid, threads_per_block](d_X, d_Y, d_Z)
    
    Z = d_Z.copy_to_host()

    return Z

@cuda.jit
def matmul_kernel(A, B, C):
    n_threads = cuda.gridsize(1)
    thread_idx = cuda.grid(1)
    ops_per_thread = int(math.ceil((A.shape[0] * B.shape[1]) / n_threads))
    init_idx = thread_idx * ops_per_thread

    for i in range(ops_per_thread):
        if C.shape[1] < C.shape[0]:
            y = (init_idx + i) % C.shape[1]
            x = int((init_idx + i) / C.shape[1])
        else:
            x = (init_idx + i) % C.shape[0]
            y = int((init_idx + i) / C.shape[0])

        res = 0.0
        for k in range(A.shape[1]):
            res += A[x, k] * B[k, y]
        C[x, y] = res

#this is the comparison function - keep it as it is, don't change X or Y.
def matmul_comparison():
    X = np.random.randn(784, 128)
    Y = np.random.randn(128, 64)
    #X = np.random.randint(0,256,(128, 64))
    #Y = np.random.randint(0,256, (64, 10))
    #X = np.random.randint(0,256,(784, 128))
    #Y = np.random.randint(0,256, (128, 64))
    #print(np.array_equal(matmul_gpu(X, Y), matmul_numba(X, Y)))
    #print(np.array_equal(np.matmul(X, Y), matmul_gpu(X, Y)))
    #import sys
    #sys.exit()
    #np.testing.assert_array_almost_equal(matmul_trivial(X, Y), matmul_numba(X, Y))
    #print('Success trivial numba')
    np.testing.assert_array_almost_equal(matmul_numba(X, Y), matmul_gpu(X, Y), decimal=4)
    print('Success numba gpu')
    import sys
    sys.exit()
    def timer(f):
        return min(timeit.Timer(lambda: f(X, Y)).repeat(3, 100))

    #print('Python:', timer(matmul_trivial)) we will not consider this since it takes infinite time :)

    print('Numpy:', timer(np.matmul))
    print('Numba:', timer(matmul_numba))
    print('CUDA:', timer(matmul_gpu))


if __name__ == '__main__':
    os.environ['NUMBAPRO_NVVM'] = '/usr/local/cuda-9.0/nvvm/lib64/libnvvm.so'
    os.environ['NUMBAPRO_LIBDEVICE'] = '/usr/local/cuda-9.0/nvvm/libdevice/'
    matmul_comparison()
