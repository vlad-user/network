import numpy as np

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
            processes.
    Parameters
    ----------
    send : numpy array
        the array of the current process
    recv : numpy array
        an array to store the result of the reduction. Of same shape as send
    comm : MPI.Comm
    """

    batch = send.shape[0] // comm.size
    indices2split = [i + batch
                     for i in range(0, send.shape[0], batch)
                     if i + batch < send.shape[0]]

    chunks = np.split(send, indices2split)

    pid = comm.rank
    prev_pid = (pid - 1 if pid - 1 >= 0 else comm.size - 1)
    next_pid = (pid + 1 if pid + 1 < comm.size else 0)

    for i in range(comm.size - 1):
        comm.send(chunks[pid], dest=next_pid)
        chunks[prev_pid] = np.add(chunks[prev_pid],
                                  comm.recv(source=prev_pid))


    chunk2send = prev_pid
    chunk2recv = (prev_pid - 1 if prev_pid -1 >= 0 else comm.size - 1)
    for i in range(comm.size - 1):
        comm.send(chunks[chunk2send], dest=next_pid)
        chunks[chunk2recv] = comm.recv(source=prev_pid)
        chunk2send = chunk2recv
        chunk2recv = (chunk2recv - 1 if chunk2recv - 1 >= 0 else comm.size - 1)
    
    recv_idx = 0
    for chunk in chunks:
        for i in chunk:
            recv[recv_idx] = i
            recv_idx += 1