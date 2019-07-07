import multiprocessing as mp

class MyQueue(object):

    def __init__(self):
        ''' Initialize MyQueue and it's members.
        '''
        self._paren_conn, self._child_conn = mp.Pipe()
        self._lock = mp.Lock()

    def put(self, msg):
        '''Put the given message in queue.

        Parameters
        ----------
        msg : object
            the message to put.
        '''
        self._lock.acquire()
        try:
            self._child_conn.send(msg)
        finally:
            self._lock.release()

    def get(self):
        '''Get the next message from queue (FIFO)
            
        Return
        ------
        An object
        '''
        return self._paren_conn.recv()

