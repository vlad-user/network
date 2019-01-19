import sys
from time import time
import numpy as np
from my_naive_allreduce import *
from my_ring_allreduce import *
from mpi4py import MPI

la_comm = MPI.COMM_WORLD
ma_rank = la_comm.Get_rank()
la_size = la_comm.Get_size()

for size in [2**12, 2**13, 2**14]:
    print("array size:", size, 'rank:', la_comm.rank)
    data = np.random.rand(size)
    res0 = np.zeros_like(data)
    res1 = np.zeros_like(data)
    res2 = np.zeros_like(data)
    la_comm.Allreduce(data, res0)

    start1 = time()
    allreduce(data,res1,la_comm)
    
    end1 = time()
    print("naive impl time:", end1-start1)
    start1 = time()
    ringallreduce(data,res2,la_comm)
    end1 = time()
    print("ring impl time:", end1-start1)
    break

    

    #assert np.allclose(res1, res2)
for size in [2**13, 2**14]:
    print("array size:", size, 'rank:', la_comm.rank)
    data = np.random.rand(size)
    res0 = np.zeros_like(data)
    res1 = np.zeros_like(data)
    res2 = np.zeros_like(data)
    la_comm.Allreduce(data, res0)

    start1 = time()
    allreduce(data,res1,la_comm)
    
    end1 = time()
    print("naive impl time:", end1-start1)
    start1 = time()
    #ringallreduce(data,res2,la_comm)
    end1 = time()
    print("ring impl time:", end1-start1)
    break
'''
data = np.ones(16)
res0 = np.zeros_like(data)
res1 = np.zeros_like(data)
res2 = np.zeros_like(data)
print('-----------------MPI.allreduce----------------')
la_comm.Allreduce(data, res0)
print('res0=', res0)

print('-----------------allreduce----------------')
allreduce(data, res1, la_comm)
print('res1=', res1)
print('-----------------ringallreduce----------------')
ringallreduce(data, res2, la_comm)
print('res2=', res2)
'''

