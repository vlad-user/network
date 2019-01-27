import itertools

import numpy as np
from mpi4py import MPI

def allreduce(send, recv, comm):
    all_arys = [np.zeros_like(send) if i != comm.rank else send
                for i in range(comm.size)]
    reduced = np.zeros_like(send)
    communications = get_communications(comm.size)

    for src, dst in communications:
        if comm.rank == src:
            comm.send(send, dest=dst)
        if comm.rank == dst:
            all_arys[src] = comm.recv(source=src)

    reduced = sum(all_arys)
    for i in range(reduced.shape[0]):
        recv[i] = reduced[i]

def get_communications(comm_size):
    return [x for x in itertools.product(*[range(comm_size), range(comm_size)])
              if x[0] != x[1]]

def allreduce_v2(send, recv, comm):
    """ Naive all reduce implementation

    Parameters
    ----------
    send : numpy array
        the array of the current process
    recv : numpy array
        an array to store the all_arys of the reduction. Of same shape as send
    comm : MPI.Comm
    """

    all_arys = []
    reduced = np.zeros_like(send)
    if comm.rank == 0:
        all_arys.append(send)

    if comm.rank != 0:
        comm.Send([send, MPI.FLOAT], dest=0)
    else:
        for source in range(1, comm.size):
            tmp = np.zeros_like(send)
            comm.Recv([tmp, MPI.FLOAT], source=source)
            all_arys.append(tmp)
        reduced = sum(all_arys)
        for dest in range(1, comm.size):
            comm.send(reduced, dest=dest)

    if comm.rank != 0:
        reduced = comm.recv(source=0)
    
    for i in range(reduced.shape[0]):
        recv[i] = reduced[i]


