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
    rank = comm.Get_rank()
    size = comm.Get_size()
    chunks = np.array_split(send, size)
    prev_pid = (rank - 1) % size
    next_pid = (rank + 1) % size
    chunk2send = rank
    chunk2recv = prev_pid

    for _ in range(comm.size - 1):
        #comm.Send([chunks[chunk2send], MPI.FLOAT], dest=next_pid)
        send_req = None
        recv_req = None
        send_req = comm.Isend(chunks[chunk2send], dest=next_pid)

        tmp = np.zeros_like(chunks[chunk2recv])
        #comm.Recv([tmp, MPI.FLOAT], source=prev_pid)
        recv_req = comm.Irecv(tmp, source=prev_pid)
        
        if send_req is not None:
            send_req.wait()
        if recv_req is not None:
            recv_req.wait()

        chunks[chunk2recv] = np.add(chunks[chunk2recv], tmp)
        chunk2send = chunk2recv
        chunk2recv = (chunk2recv - 1) % size

    chunk2send = next_pid
    chunk2recv = (chunk2send - 1) % size

    for _ in range(comm.size - 1):
        send_req = None
        recv_req = None
        #comm.Send([chunks[chunk2send], MPI.FLOAT], dest=next_pid)
        send_req = comm.Isend(chunks[chunk2send], dest=next_pid)
        tmp = np.zeros_like(chunks[chunk2recv])
        #comm.Recv([tmp, MPI.FLOAT], source=prev_pid)
        recv_req = comm.Irecv(tmp, source=prev_pid)
        if send_req is not None:
            send_req.wait()
        if recv_req is not None:
            recv_req.wait()
        np.copyto(chunks[chunk2recv], tmp)
        #chunks[chunk2recv] = tmp
        chunk2send = chunk2recv
        chunk2recv = (chunk2recv - 1) % size
    
    np.copyto(recv, np.concatenate(chunks, axis=0))
    #recv_idx = 0
    #for chunk in chunks:
    #    for i in range(chunk.shape[0]):
    #        recv[recv_idx] = chunk[i]
    #        recv_idx += 1
    
    
