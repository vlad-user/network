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
    data = np.float32(np.random.rand(size))
    res0 = np.zeros_like(data)
    res1 = np.zeros_like(data)
    res2 = np.zeros_like(data)
    res0 = la_comm.allreduce(data)

    start1 = time()
    allreduce(data, res1, la_comm)
    assert np.allclose(res0, res1)
    np.testing.assert_almost_equal(res0, res1, decimal=5)
    end1 = time()
    print("naive impl time:", end1-start1)
    start1 = time()
    ringallreduce(data,res2,la_comm)
    end1 = time()
    print("ring impl time:", end1-start1)
    np.testing.assert_almost_equal(res0, res2, decimal=5)
    assert np.allclose(res0, res2)
    np.testing.assert_almost_equal(res1, res2, decimal=5)


'''
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
'''
verbose = False
import random
#data = np.float32(np.random.randn(2**14))
#data = np.float32([random.choice([0, 1]) for _ in range(2**3)])
data = np.float32(np.ones(12))
data[data.shape[0]//2] = la_comm.rank

print(f'My rank is {la_comm.rank} and my data is:\n{data}')

res0 = np.zeros_like(data)
res1 = np.zeros_like(data)
res2 = np.zeros_like(data)
#print('-----------------MPI.allreduce----------------', la_comm.rank)
#la_comm.Allreduce([data, MPI.FLOAT], [res0, MPI.FLOAT], op=MPI.SUM)
res0 = la_comm.allreduce(data)
#print('res0=', res0)

if verbose: print('-----------------allreduce----------------', la_comm.rank)
allreduce(data, res1, la_comm)
if verbose: print('res1=', res1)
np.testing.assert_almost_equal(res0, res1, decimal=5)

if verbose: print('-----------------ringallreduce----------------', la_comm.rank)
ringallreduce(data, res2, la_comm)
if verbose: print('res2=', res2)

if la_comm.rank == 0:
    import pickle
    with open('res0.pkl', 'wb') as fo:
        pickle.dump(res0, fo, protocol=pickle.HIGHEST_PROTOCOL)
    with open('res2.pkl', 'wb') as fo:
        pickle.dump(res2, fo, protocol=pickle.HIGHEST_PROTOCOL)
    with open('res1.pkl', 'wb') as fo:
        pickle.dump(res1, fo, protocol=pickle.HIGHEST_PROTOCOL)

np.testing.assert_almost_equal(res0, res2, decimal=5)
if verbose: print('------------------------------------------------------------------------------------')
'''
