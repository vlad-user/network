import random
import multiprocessing as mp
from time import sleep

from scipy import ndimage
import numpy as np

IMG_SIZE = 28

class Worker(mp.Process):
    
    def __init__(self, jobs, result, training_data=None, batch_size=None):
        super().__init__(target=self.run)
        ''' Initialize Worker and it's members.

        Parameters
        ----------
        jobs: Queue
            A jobs Queue for the worker.
        result: Queue
            A results Queue for the worker to put it's results in.
        
        You should add parameters if you think you need to.
        '''

        self._jobs = jobs
        self._result = result
        self._training_data = training_data
        self._batch_size = batch_size
    
    @staticmethod
    def image_to_flatten(x):
        """Flattens array `x`.

        Args:
            `x`: A numpy array to be flattened.

        Returns:
            A flattened array.
        """

        return np.reshape(x, np.prod(np.asarray(x.shape)))

    @staticmethod
    def flatten_to_image(x):
        """Convertx a flatten image `x` into 2d image."""

        return np.reshape(x, (IMG_SIZE, IMG_SIZE))

    @staticmethod
    def rotate(image, angle):
        '''Rotate given image to the given angle

        Parameters
        ----------
        image : numpy array
            An array of size 784 of pixels
        angle : int
            The angle to rotate the image
            
        Return
        ------
        An numpy array of same shape
        '''

        res = Worker.image_to_flatten(ndimage.rotate(
            Worker.flatten_to_image(image), angle, reshape=False))

        return res
        
    @staticmethod
    def shift(image, dx, dy):
        '''Shift given image

        Parameters
        ----------
        image : numpy array
            An array of shape 784 of pixels
        dx : int
            The number of pixels to move in the x-axis
        dy : int
            The number of pixels to move in the y-axis
            
        Return
        ------
        An numpy array of same shape
        '''
        '''
        return ndimage.shift(image,
                             (dx, dy),
                             mode='constant',
                             cval=0.0)
        '''

        reshaped_img = Worker.flatten_to_image(image)
        if dx != 0:
            reshaped_img = np.roll(reshaped_img, -dx, axis=1)
        if dy != 0:
            reshaped_img = np.roll(reshaped_img, -dy, axis=0)

        if dy > 0:
            reshaped_img[-dy:, :] = 0.0
        elif dy < 0:
            reshaped_img[:-dy, :] = 0.0

        if dx > 0:
            reshaped_img[:, -dx:] = 0.0
        elif dx < 0:
            reshaped_img[:, :-dx] = 0.0

        res = Worker.image_to_flatten(reshaped_img)

        return res
    
    @staticmethod
    def step_func(image, steps):
        '''Transform the image pixels acording to the step function

        Parameters
        ----------
        image : numpy array
            An array of shape 784 of pixels
        steps : int
            The number of steps between 0 and 1

        Return
        ------
        An numpy array of same shape
        '''

        res = np.vectorize(lambda v: (1/(steps - 1))*np.floor(steps*v))(image)

        return res

    @staticmethod
    def skew(image, tilt):
        '''Skew the image

        Parameters
        ----------
        image : numpy array
            An array of size 784 of pixels
        tilt : float
            The skew paramater

        Return
        ------
        An numpy array of same shape
        '''

        reshaped_image = Worker.flatten_to_image(image)
        skewed_image = np.zeros(shape=reshaped_image.shape)

        for x in range(skewed_image.shape[0]):
            for y in range(skewed_image.shape[1]):
                skewed_image[y][x] = (0.0
                                      if (int(x + y*tilt) >= skewed_image.shape[1])
                                      else reshaped_image[y][int(x+y*tilt)])

        res = Worker.image_to_flatten(skewed_image)
        return res

    def process_image(self, image):
        '''Apply the image process functions

        Parameters
        ----------
        image: numpy array
            An array of size 784 of pixels

        Return
        ------
        An numpy array of same shape
        '''

        funcs_and_args = [(self.skew, (np.random.uniform(low=-0.04, high=0.04), )),
                          (self.step_func, (random.randint(130, 255), )),
                          (self.shift, (random.randint(-2, 2), random.randint(-1, 1))),
                          (self.rotate, (random.randint(-10, 10), ))]

        random.shuffle(funcs_and_args)

        for func, args in funcs_and_args:
            image = (func(image, args[0])
                     if len(args) == 1
                     else func(image, args[0], args[1]))

        return image

    def run(self):
        '''Process images from the jobs queue and add the result to the result queue.
        '''
        while True:

            image, label = self.get_jobs_queue().get()
            processed = self.process_image(image)
            self.get_result_queue().put((processed, label)) 

    def get_jobs_queue(self):
        return self._jobs

    def get_result_queue(self):
        return self._result