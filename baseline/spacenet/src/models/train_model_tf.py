"""
CS4793

@authors: Melanie Bischoff, Malay Bhakta, Jordan Johnson
"""

import sys
import platform
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

"""
Goal: 
   
Description: 
    
"""
if __name__ == "__main__":
    print("Part A - 8 input nodes, 3 hidden nodes, and 8 output nodes")

    # Set a random seed so runs are repeatable
    np.random.seed(98234)  # need to set numpy seed since np.random.shuffle is used
    tf.random.set_seed(98234)  # and tensorflow graph seed

    #Start doing some work....
    model = keras.Sequential()
