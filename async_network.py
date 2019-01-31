from network import *
import itertools
import sys
import numpy as np
import math
import  mpi4py
from time import time
mpi4py.rc(initialize=False, finalize=False)
from mpi4py import MPI


class AsynchronicNeuralNetwork(NeuralNetwork):

    def __init__(self, sizes=list(), learning_rate=1.0, mini_batch_size=16, number_of_batches=16,
                 epochs=10, number_of_masters = 1,  matmul=np.matmul):
    # calling super constructor
        super().__init__(sizes, learning_rate, mini_batch_size, number_of_batches, epochs, matmul)
    # setting number of workers and masters
        self.num_masters = number_of_masters

    def fit(self, training_data, validation_data=None):
        # MPI setup
        if not MPI.Is_initialized():
            MPI.Init()
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.num_workers = self.size - self.num_masters

        
        self.layers_per_master = self.num_layers // self.num_masters
        
        # split up work
        if self.rank < self.num_masters:
            self.do_master(validation_data)
        else:
            self.do_worker(training_data)
        
        # when all is done
        self.comm.Barrier()
        MPI.Finalize()
    
    def do_worker(self, training_data):
    #worker functionality
    
    #Parameters
    #    ----------
    #training_data : a tuple of data and labels to train the NN with
    
        
    # setting up the number of batches the worker should do every epoch
        #TODO
        # we want that worker i will do mini_batches[t] if t%i == 0.
        my_worker_index = self.rank - self.num_masters # my index as if it was started from 0
        for epoch in range(self.epochs):  
            # creating batches for epoch
            data = training_data[0]
            labels = training_data[1]
            mini_batches = self.create_batches(data, labels, self.mini_batch_size)
            for idx in range(my_worker_index, self.number_of_batches, self.num_workers):
                x, y = mini_batches[idx]
                # do work - don't change this
                self.forward_prop(x)
                nabla_b, nabla_w = self.back_prop(y)
                
                # send nabla_b, nabla_w to masters 
                #TODO
                send_list = []
                # the worker sends all his layers
                for i in range(self.num_layers):
                    send_list.append(self.comm.Isend(nabla_w[i], dest = i%self.num_masters))
                    send_list.append(self.comm.Isend(nabla_b[i], dest = i%self.num_masters))
                # for request in send_list:
                #   MPI.Request.Wait(request)
                # recieve new self.weight and self.biases values from masters
                # TODO
                recv_list = []
                # the worker receives new weights & biases from all masters
                for i in range(self.num_layers):
                    recv_list.append(self.comm.Irecv(self.weights[i], source = i%self.num_masters))
                    recv_list.append(self.comm.Irecv(self.biases[i], source = i%self.num_masters))
                for request in recv_list:
                    MPI.Request.Wait(request)
                
                             
    def do_master(self, validation_data):
        ''' master functionality
        
    Parameters
        ----------
    validation_data : a tuple of data and labels to train the NN with
    '''
        
    #setting up the layers this master does
        nabla_w = []
        nabla_b = []
        for i in range(self.rank, self.num_layers, self.num_masters):
            nabla_w.append(np.zeros_like(self.weights[i]))
            nabla_b.append(np.zeros_like(self.biases[i]))
            
        for epoch in range(self.epochs ):
            for batch in range(self.number_of_batches):
                
        # wait for any worker to finish batch and
        # get the nabla_w, nabla_b for the master's layers
        # TODO
                recv_list = []
                glob_status = MPI.Status()
                # as we send the layers "sequentially", we first scan to the first location
                # in nabla array , then according to the worker who sent the layers,
                # we keep receiving the regarding layers
                req_to_get_worker = self.comm.Irecv(nabla_w[0],source = MPI.ANY_SOURCE)
                MPI.Request.Wait(req_to_get_worker, glob_status)
                # now we can get the source and stuff
                worker_to_receive_from = glob_status.Get_source()
                #get biases for first place
                req_to_get_worker2 = self.comm.Irecv(nabla_b[0],source=worker_to_receive_from)                
                MPI.Request.Wait(req_to_get_worker2)
                for i in range(1,len(nabla_w)):
                    recv_list.append(self.comm.Irecv(nabla_w[i],source = worker_to_receive_from))
                    recv_list.append(self.comm.Irecv(nabla_b[i],source = worker_to_receive_from))
                for request in recv_list:
                    MPI.Request.Wait(request)
                # calculate new weights and biases (of layers in charge)
                for i, dw, db in zip(range(self.rank, self.num_layers, self.num_masters), nabla_w, nabla_b):
                    self.weights[i] = self.weights[i] - self.eta *  dw
                    self.biases[i] = self.biases[i] - self.eta *  db
            
                # send new values (of layers in charge)
                #TODO
                send_list = []
                for i in range(self.rank, self.num_layers, self.num_masters):
                    send_list.append(self.comm.Isend(self.weights[i],dest = worker_to_receive_from))
                    send_list.append(self.comm.Isend(self.biases[i],dest = worker_to_receive_from))
                # for request in  send_list:
                #   MPI.Request.Wait(request)
                
            self.print_progress(validation_data, epoch)

        # gather relevant weight and biases to process 0
        #TODO
        if self.rank != 0:
            for i in range(self.rank, self.num_layers, self.num_masters):
                req_w = self.comm.Isend(self.weights[i], dest = 0)
                # MPI.Request.Wait(req_w)
                req_b = self.comm.Isend(self.biases[i], dest = 0)
                # MPI.Request.Wait(req_b)
        if self.rank == 0:
            recv_list = []
            for master in range(1,self.num_masters):
                for layer in range(master, self.num_layers,self.num_masters):
                    recv_list.append(self.comm.Irecv(self.weights[layer],source = master))
                    recv_list.append(self.comm.Irecv(self.biases[layer],source = master))
            for request in recv_list:
                MPI.Request.Wait(request)
