import sys
import pickle

from network import *
from my_ring_allreduce import *
import utils
import  mpi4py
mpi4py.rc(initialize=False, finalize=False)
from mpi4py import MPI


class SynchronicNeuralNetwork(NeuralNetwork):

    def fit(self, training_data, validation_data=None):
        if not MPI.Is_initialized():
            MPI.Init()
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        nabla_w = []
        nabla_b = []
        for epoch in range(self.epochs):
            data = training_data[0]
            labels = training_data[1]
            mini_batches = self.create_batches(data, labels, self.mini_batch_size // size)
            #print(f'My rank is {rank} and my len of mini_batches is: {len(mini_batches)}')
            print('My rank is', rank, '/', size)
            chunks = self._chunkify(mini_batches, len(mini_batches) // size)
            #print(f'My rank is {rank} and my len of    chunks    is: {len(chunks)}')
            for x, y in chunks[rank]:
                # doing props

                self.forward_prop(x)
                ma_nabla_b, ma_nabla_w = self.back_prop(y)
                for b in ma_nabla_b:
                    tmp = np.zeros_like(b)
                    comm.Allreduce(b, tmp)
                    #ringallreduce(b, tmp, comm)
                    nabla_b.append(tmp)
                for w in ma_nabla_w:
                    tmp = np.zeros_like(w)
                    comm.Allreduce(w, tmp)
                    #ringallreduce(w, tmp, comm)
                    nabla_w.append(tmp)

                #calculate work
                self.weights = [w - self.eta * dw for w, dw in zip(self.weights, nabla_w)]
                self.biases = [b - self.eta * db for b, db in zip(self.biases, nabla_b)]
                nabla_w = []
                nabla_b = []    
            
            self.print_progress(validation_data, epoch)

        MPI.Finalize()

    def _chunkify(self, list_, n_chunks):
        return [list_[i:i+n_chunks] for i in range(0, len(list_), n_chunks)]