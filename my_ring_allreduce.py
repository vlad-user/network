import numpy as np
from mpi4py import MPI

""" Implementation of a ring-reduce with addition. """
def ringallreduce(send, recv, comm):
    """ ring all reduce implementation
    You need to use the algorithm shown in the lecture.

    Algorithm:
        1. Init: For each process in communicator `comm`,
            divide the array to be reduced `send` into
            `comm.size` chunks.
        2. Repeat `comm.size - 1` times:    
            For each process `p`, do:
                * send `chunk[p]` to the process `p+1 mod comm.size`.
                * receive `chunk[p-1]` from process `p-1`.
                * perform reduction on the received `chunk[p-1]`
                    and on its own chunk[p-1]
                * send the reduced chunk to the next process `p+1`.
        3. Similar to step 2, transmit reduced chunks to all of the
            processes withtout reduction.
    Parameters
    ----------
    send : numpy array
        the array of the current process
    recv : numpy array
        an array to store the result of the reduction. Of same shape as send
    comm : MPI.Comm
    """
    chunks = np.array_split(send, comm.size)
    prev_pid = (comm.rank - 1) % comm.size
    next_pid = (comm.rank + 1) % comm.size
    chunk2send = comm.rank
    chunk2recv = prev_pid

    for _ in range(comm.size - 1):
        comm.send(chunks[chunk2send], dest=next_pid)
        chunks[chunk2recv] = np.add(chunks[chunk2recv], comm.recv(source=prev_pid))
        chunk2send = chunk2recv
        chunk2recv = (chunk2recv - 1) % comm.size

    chunk2send = next_pid
    chunk2recv = (chunk2send - 1) % comm.size

    for _ in range(comm.size - 1):
        comm.send(chunks[chunk2send], dest=next_pid)
        chunks[chunk2recv] = comm.recv(source=prev_pid)
        chunk2send = chunk2recv
        chunk2recv = (chunk2recv - 1) % comm.size
    
    recv_idx = 0
    for chunk in chunks:
        for i in range(chunk.shape[0]):
            recv[recv_idx] = chunk[i]
            recv_idx += 1
    
    
