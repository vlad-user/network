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
        MPI.Init()
        comm = MPI.COMM_WORLD
        sendbuff = []
        nabla_w = []
        nabla_b = []
        for epoch in range(self.epochs):
            data = training_data[0]
            labels = training_data[1]
            mini_batches = self.create_batches(data, labels, self.mini_batch_size // comm.size)

            for x, y in mini_batches:
                # doing props
                if comm.rank == 0:
                    sendbuff = utils.create_batches(x, y, len(x)//comm.size)

                (x_, y_) = comm.scatter(sendbuff, root=0)

                self.forward_prop(x_)
                ma_nabla_b, ma_nabla_w = self.back_prop(y_)
                
                #nabla_w = []
                #nabla_b = []

                #comm.Allreduce(ma_nabla_w, nabla_w)
                #comm.Allreduce(ma_nabla_b, nabla_b)
                #nabla_w = [x[0] for x in recvbuff]
                #nabla_b = [x[1] for x in recvbuff]
                
                recvbuff = comm.allgather((ma_nabla_w, ma_nabla_b))
                # summing all ma_nabla_b and ma_nabla_w to nabla_w and nabla_b
                
                #if comm.rank == 0:
                nabla_w_list = [x[0] for x in recvbuff]
                nabla_b_list = [x[1] for x in recvbuff]

                nabla_w = [sum(x) for x in zip(*nabla_w_list)]
                nabla_b = [sum(x) for x in zip(*nabla_b_list)]
                
                #calculate work
                self.weights = [w - self.eta * dw for w, dw in zip(self.weights, nabla_w)]
                self.biases = [b - self.eta * db for b, db in zip(self.biases, nabla_b)]        
            if comm.rank == 0:
                self.print_progress(validation_data, epoch)

        MPI.Finalize()
