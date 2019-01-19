import numpy as np

def allreduce(send, recv, comm):
    """ Naive all reduce implementation

    Parameters
    ----------
    send : numpy array
        the array of the current process
    recv : numpy array
        an array to store the result of the reduction. Of same shape as send
    comm : MPI.Comm
    """
    result = []
    for dest in range(comm.size):
        if dest != comm.rank:
            comm.send(send, dest=dest)

    for source in range(comm.size):
        if source != comm.rank:
            result.append(comm.recv(source=source))
    
    result.append(send)
    for i, x in enumerate(zip(*result)):
        recv[i] = sum(x)


