import sys
import multiprocessing as mp

from network import *
from preprocessor import Worker
import utils
from my_queue import MyQueue

class IPNeuralNetwork(NeuralNetwork):
    
    def fit(self, training_data, validation_data=None):
        '''
        Override this function to create and destroy workers
        '''

        self._jobs_queue = mp.Queue()
        self._res_queue = MyQueue()
        n_cpus = mp.cpu_count()
        self._processes = [Worker(self._jobs_queue, self._res_queue)
                           for i in range(n_cpus)]
        for p in self._processes:
            p.start()

        try:
            super().fit(training_data, validation_data)
        except:
            for p in self._processes:
                p.terminate()
            raise
        
        for p in self._processes:
            p.terminate()
        
        
    
    def create_batches(self, data, labels, batch_size):
        '''
        Override this function to return batches created by workers
        '''
        n_samples = len(data)

        for d, l in zip(data, labels):
            self._jobs_queue.put((d, l))

        result = []

        while len(result) != n_samples:
            result.append(self._res_queue.get())

        data = np.asarray([r[0] for r in result])
        labels = np.asarray([r[1] for r in result])

        return super().create_batches(data, labels, batch_size)



    
