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
        
        for epoch in range(self.epochs):
            
            data = training_data[0]
            labels = training_data[1]
            mini_batches = self.create_batches(data, labels, self.mini_batch_size // size)
            
            for x, y in mini_batches:
                
                self.forward_prop(x)
                ma_nabla_b, ma_nabla_w = self.back_prop(y)
                
                nabla_w = [np.zeros_like(w) for w in ma_nabla_w]
                nabla_b = [np.zeros_like(b) for b in ma_nabla_b]

                _ = [comm.Allreduce(w, reduced, op=MPI.SUM)
                     for (w, reduced) in zip(ma_nabla_w, nabla_w)]
                _ = [comm.Allreduce(b, reduced, op=MPI.SUM)
                     for (b, reduced) in zip(ma_nabla_b, nabla_b)]

                self.weights = [w - self.eta * dw for w, dw in zip(self.weights, nabla_w)]
                self.biases = [b - self.eta * db for b, db in zip(self.biases, nabla_b)]        
                
            self.print_progress(validation_data, epoch)
            sys.stdout.flush()

        MPI.Finalize()

    def fit_v2(self, training_data, validation_data=None):
        if not MPI.Is_initialized():
            MPI.Init()
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        for epoch in range(self.epochs):
            data = training_data[0]
            labels = training_data[1]
            mini_batches = self.create_batches(data, labels, self.mini_batch_size // size)
            
            for x, y in mini_batches:
                self.forward_prop(x)
                ma_nabla_b, ma_nabla_w = self.back_prop(y)
                nabla_w = [np.zeros_like(w) for w in ma_nabla_w]
                nabla_b = [np.zeros_like(b) for b in ma_nabla_b]

                _ = (comm.Allreduce(w, reduced, op=MPI.SUM)
                     for (w, reduced) in zip(ma_nabla_w, nabla_w))
                _ = (comm.Allreduce(b, reduced, op=MPI.SUM)
                     for (b, reduced) in zip(ma_nabla_b, nabla_b))

                assert any(np.array_equal(x, np.zeros_like(x)) for x in nabla_w + nabla_b)
                self.weights = [w - self.eta * dw for w, dw in zip(self.weights, nabla_w)]
                self.biases = [b - self.eta * db for b, db in zip(self.biases, nabla_b)]
            self.print_progress(validation_data, epoch)
            sys.stdout.flush()
            '''
            chunks = self._chunkify(mini_batches, size)
            for x, y in chunks[rank]:

                # doing props
                self.forward_prop(x)
                ma_nabla_b, ma_nabla_w = self.back_prop(y)
                for b in ma_nabla_b:
                    tmp = np.zeros_like(b)
                    comm.Allreduce(b, tmp)

                    nabla_b.append(tmp)
                for w in ma_nabla_w:
                    tmp = np.zeros_like(w)
                    comm.Allreduce(w, tmp)

                    nabla_w.append(tmp)

                #calculate work
                self.weights = [w - self.eta * dw for w, dw in zip(self.weights, nabla_w)]
                self.biases = [b - self.eta * db for b, db in zip(self.biases, nabla_b)]
                nabla_w = []
                nabla_b = []    
            
            self.print_progress(validation_data, epoch)
            sys.stdout.flush()
            '''
        MPI.Finalize()

    def _chunkify(self, lst, n):
        return [lst[i::n] for i in range(n)]
