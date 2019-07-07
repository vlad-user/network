from numba import cuda
from numba import njit
import numba
import numpy as np


def convolution_gpu(kernel, image):
    '''Convolve using gpu
    Parameters
    ----------
    kernel : numpy array
        A small matrix
    image : numpy array
        A larger matrix of the image pixels
            
    Return
    ------
    An numpy array of same shape as image
    '''
    kernel = np.ascontiguousarray(np.flipud(np.fliplr(kernel)))
    kernel_size_x = kernel.shape[0]
    kernel_size_y = kernel.shape[1]
    padding_x = int((kernel_size_x - 1)/2)
    padding_y = int((kernel_size_y - 1)/2)

    padded = np.zeros((image.shape[0] + kernel_size_x - 1,
                       image.shape[1] + kernel_size_y - 1))

    padded[padding_x:-padding_x, padding_y:-padding_y] = image
    
    result = np.zeros(shape=image.shape)

    d_kernel = cuda.to_device(kernel)
    d_padded = cuda.to_device(padded)
    d_result = cuda.to_device(result)

    threads_per_block = 1024
    blocks_per_grid = int((image.shape[0]*image.shape[1])/threads_per_block) + 1
    conv2d_gpu_kernel[blocks_per_grid, threads_per_block](
            d_kernel, d_padded, d_result)

    result = d_result.copy_to_host()

    return result

@cuda.jit
def conv2d_gpu_kernel(kernel, padded, result):
    """Calculates single 2d convolution operation on GPU.
    
    Args:
        `kernel`: A convolution kernel - two dimensional numpy array.
        `padded`: Original image padded with zeros resulting in shape
            `(image.shape[0] + kernel.shape[0] - 1,
             image.shape[1] + kernel.shape[1] - 1)`.
        `result`: An array of zeros of same shape as the orignal image.
    """

    thread_idx = cuda.grid(1)

    img_size_x = padded.shape[0] - kernel.shape[0] + 1
    img_size_y = padded.shape[1] - kernel.shape[1] + 1

    if thread_idx < img_size_x*img_size_y:
        
        x = int(thread_idx/img_size_x)
        y = img_size_y - ((x + 1)*img_size_x - thread_idx)

        res = 0.0
        for kx in range(kernel.shape[0]):
            for ky in range(kernel.shape[1]):
                res += kernel[kx, ky]*padded[x+kx, y+ky]

        result[x, y] = res

@njit(parallel=True, fastmath=True)
def convolution_numba(kernel, image):
    '''Convolve using numba
    Parameters
    ----------
    kernel : numpy array
        A small matrix
    image : numpy array
        A larger matrix of the image pixels
            
    Return
    ------
    An numpy array of same shape as image
    '''

    kernel_ = np.zeros(shape=kernel.shape)
    # flip horizontally
    for x in range(kernel.shape[0]):
        for y in range(kernel.shape[1]):
            kernel_[x, kernel.shape[1] - y - 1] = kernel[x, y]
    # flip vertically
    for y in range(kernel.shape[1]):
        for x in range(kernel.shape[0]):
             kernel[kernel.shape[0] - x - 1, y] = kernel_[x, y]

    padding_x = int((kernel.shape[0] - 1)/2)
    padding_y = int((kernel.shape[1] - 1)/2)

    padded = np.zeros((image.shape[0] + kernel.shape[0] - 1,
                       image.shape[1] + kernel.shape[1] - 1))
    padded[padding_x:-padding_x, padding_y:-padding_y] = image

    res = np.zeros(shape=image.shape)

    for y in numba.prange(image.shape[1]):
        for x in numba.prange(image.shape[0]):
            res[x, y] = np.sum(kernel*padded[x:x+kernel.shape[0], y:y+kernel.shape[1]])
        
    return res
