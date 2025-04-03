# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 11:47:44 2025

@author: Nicola
"""

# import os

# # Set number of CPU threads before importing TensorFlow
# num_threads = os.cpu_count()  # Get total available CPU cores
# os.environ["OMP_NUM_THREADS"] = str(num_threads)
# os.environ["TF_NUM_INTEROP_THREADS"] = str(num_threads)
# os.environ["TF_NUM_INTRAOP_THREADS"] = str(num_threads)


import numpy as np
from tensorflow import keras
import tensorflow as tf 

# from keras.models import Sequential
from keras.layers import Activation, Dense, Input, BatchNormalization
from keras.optimizers import RMSprop, Adam, SGD
from keras.callbacks import LearningRateScheduler
from keras.initializers import GlorotUniform, GlorotNormal
from scipy.stats import qmc

# # Enable TensorFlow to use all available CPU cores
# num_threads = os.cpu_count()  # Get the number of available CPU cores
# tf.config.threading.set_intra_op_parallelism_threads(num_threads)
# tf.config.threading.set_inter_op_parallelism_threads(num_threads)
# tf.config.optimizer.set_jit(True)  # Enable XLA optimization

# Enable XLA Optimization
# tf.config.optimizer.set_jit(True)

def NN_generator(InputShape, nLayers, nNodes, activationFun='relu', loss='mse', optimizer='rmsprop', LR=0.001, lr_scheduler=None, BatchNorm=0, show=1):
    
    # First Layer : Input Layer
    input_layer = Input(shape=InputShape)
    
    # Defining the first hidden layer
    x = Dense(nNodes, activation=activationFun)(input_layer)
    
    if BatchNorm==1:
        
        x = BatchNormalization()(x)
    
    # adding more layers
    
    i = 1
    
    while i<nLayers:
        
        x = Dense(nNodes, activation=activationFun)(x)
        if BatchNorm==1:
            x = BatchNormalization()(x)
        i+=1
    
    # adding the last layer
    x = Dense(1, activation='linear')(x)
    
    model = keras.models.Model(inputs=input_layer, outputs=x)
    
    # # Compiling the model 
    
    # Configuring the optimizer with the specified learning rate
    if optimizer == 'rmsprop':
        opt = RMSprop(learning_rate=LR)
    elif optimizer == 'adam':
        opt = Adam(learning_rate=LR)
    elif optimizer == 'sgd':
        opt = SGD(learning_rate=LR)
    else:
        raise ValueError("Unsupported optimizer. Choose 'rmsprop', 'adam', or 'sgd'.")
    
    # Compiling the model
    model.compile(loss=loss, optimizer=opt)
    
    # ## Different Optimizers are compared in companion notebook
    # model.compile(loss=loss, optimizer=optimizer)
    
    if show==1:
        print(model.summary())
        
    return model
    

def ParamGenerator(n, nParams, l_bounds, u_bounds):
    
    sampler = qmc.LatinHypercube(d=nParams)
    unif_sample = sampler.random(n)
    
    gen_params = qmc.scale(unif_sample, l_bounds, u_bounds)
    
    return gen_params
    
