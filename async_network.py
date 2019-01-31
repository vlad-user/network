from network import *
import itertools
import sys
import numpy as np
import math
import mpi4py
from time import time

mpi4py.rc(initialize=False, finalize=False)
from mpi4py import MPI


class AsynchronicNeuralNetwork(NeuralNetwork):

    def __init__(self, sizes=list(), learning_rate=1.0, mini_batch_size=16, number_of_batches=16,
                 epochs=10, number_of_masters=1, matmul=np.matmul):
        # calling super constructor
        super().__init__(sizes, learning_rate, mini_batch_size, number_of_batches, epochs, matmul)
        # setting number of masters
        self.num_masters = number_of_masters
        # We want to divide the layers between the masters so let's define:
        self.layers_per_master = self.num_layers / self.num_masters
        # print("init done")

    def fit(self, training_data, validation_data=None):
        # MPI setup
        if not MPI.Is_initialized():
            MPI.Init()
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        # setting number of workers
        self.num_workers = self.size - self.num_masters

        self.layers_per_master = self.num_layers // self.num_masters

        # split up work
        # print("split up!")
        if self.rank < self.num_masters:
            self.do_master(validation_data)
        else:
            self.do_worker(training_data)

        # when all is done
        self.comm.Barrier()
        MPI.Finalize()

    def do_worker(self, training_data):
        '''
        worker functionality

        Parameters
            ----------
        training_data : a tuple of data and labels to train the NN with
        '''

        # setting up the number of batches the worker should do every epoch
        # TODO
        # print("Worker " + str(self.rank))
        for epoch in range(self.epochs):
            # creating batches for epoch
            data = training_data[0]
            labels = training_data[1]
            # We want all of the workers together to do number_of_batches batches. Currently every
            # worker does that so we want every worker to do number_of_batches/number_of_workers batches
            res_batches = self.create_batches(data, labels, self.mini_batch_size)
            mini_batches = res_batches[0:int(self.number_of_batches / self.num_workers)]
            num_layers_multiple = self.num_masters*self.num_layers
            for x, y in mini_batches:
                # do work - don't change this
                self.forward_prop(x)
                nabla_b, nabla_w = self.back_prop(y)

                # send nabla_b, nabla_w to masters
                for i in range(0, self.num_layers):
                    curr_master = i%self.num_masters
                    self.comm.Isend(nabla_b[i], curr_master, 2*i)
                    self.comm.Isend(nabla_w[i], curr_master, 2*i+1)

                #We'll need to check which answeres did we get
                recv_requests = []
                for i in range(0, self.num_layers):
                    expected_master =  i%self.num_masters
                    request_bias   = self.comm.Irecv(self.biases[i], expected_master, 2*i)
                    request_weight = self.comm.Irecv(self.weights[i], expected_master, 2*i+1)

                    recv_requests.append(request_bias)
                    recv_requests.append(request_weight)
                for x in recv_requests:
                    x.Wait()
                #make sure it all came and start over

    def do_master(self, validation_data):
        ''' master functionality

        Parameters
        ----------
        validation_data : a tuple of data and labels to train the NN with
        '''

        # setting up the layers this master does
        # print("Master " + str(self.rank))
        nabla_w = []
        nabla_b = []
        for i in range(self.rank, self.num_layers, self.num_masters):
            nabla_w.append(np.zeros_like(self.weights[i]))
            nabla_b.append(np.zeros_like(self.biases[i]))

        for epoch in range(self.epochs):
            for batch in range(self.number_of_batches):
                # print("Batch number " + str(batch) + " in " + str(self.rank))

                # wait for any worker to finish batch and
                # get the nabla_w, nabla_b for the master's layers
                status = MPI.Status()
                worker_request = self.comm.Irecv(nabla_b[0], MPI.ANY_SOURCE, 2*self.rank)
                worker_request.Wait(status)
                worker = status.Get_source()
                # print("Master " + str(self.rank) + " learned the source of " + str(worker))

                recv_requests = []
                for c, x in enumerate(range(self.rank, self.num_layers, self.num_masters)):
                    if not c==0:
                        request_bias = self.comm.Irecv(nabla_b[c], worker, 2*x)
                        recv_requests.append(request_bias)
                    request_weight = self.comm.Irecv(nabla_w[c], worker, 2 * x+1)
                    recv_requests.append(request_weight)
                # print("Master "+ str(self.rank) + " got all of the requests, now waiting")
                for x in recv_requests:
                    x.Wait()

                # print("Master " + str(self.rank) + " is done waiting")
                # calculate new weights and biases (of layers in charge)
                for i, dw, db in zip(range(self.rank, self.num_layers, self.num_masters), nabla_w, nabla_b):
                    self.weights[i] = self.weights[i] - self.eta * dw
                    self.biases[i] = self.biases[i] - self.eta * db
                # print("Master " + str(self.rank) + " TRAINED")

                for c, x in enumerate(range(self.rank, self.num_layers, self.num_masters)):
                    self.comm.Isend(self.biases[x], worker, 2*x)
                    self.comm.Isend(self.weights[x], worker, 2*x+1)
                # send new values (of layers in charge)

                # print("Master " + str(self.rank) + " sent back to worker " + str(worker))
                nabla_w = []
                nabla_b = []
                for i in range(self.rank, self.num_layers, self.num_masters):
                    nabla_w.append(np.zeros_like(self.weights[i]))
                    nabla_b.append(np.zeros_like(self.biases[i]))

            # TODO

            self.print_progress(validation_data, epoch)

        # gather relevant weight and biases to process 0
        if not self.rank==0:
            for c, x in enumerate(range(self.rank, self.num_layers, self.num_masters)):
                self.comm.Isend(self.biases[x], 0, 2 * x)
                self.comm.Isend(self.weights[x], 0, 2 * x + 1)
        else:
            requests = []
            for master in range(1, self.num_masters):
                for x in range(master, self.num_layers, self.num_masters):
                    master_bias_request = self.comm.Irecv(self.biases[x], master, 2 * x)
                    master_weight_request = self.comm.Irecv(self.weights[x], master, 2 * x + 1)

                    requests.append(master_bias_request)
                    requests.append(master_weight_request)
            for x in requests:
                x.Wait()
