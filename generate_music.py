from __future__ import print_function
import IPython
import sys
from music21 import *
import numpy as np
from grammar import *
from qa import *
from preprocess import * 
from music_utils import *
from data_utils import *
from keras.models import load_model, Model
from keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
from keras.initializers import glorot_uniform
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras import backend as K

IPython.display.Audio('./data/30s_seq.mp3')

#Load data
# X: (m, Tx, 78): m trainning example, each of which is a snippet of Tx = 30 musical values, each value has 78 different posible values, represented  as a one-hot vector.
# Y: (Ty, m, 78)
# n_values: The number of unique values in this dataset
# indices_value: python dictionary mapping from 0-77 to musical values

X, Y, n_values, indices_values = load_music_utils()
print('shape of X:', X.shape)
print('number of training examples:', X.shape[0])
print('Tx (length of sequence):', X.shape[1])
print('total # of unique values:', n_values)
print('Shape of Y:', Y.shape)

# Define object layer

n_a = 64 
reshapor = Reshape((1, 78))                        
LSTM_cell = LSTM(n_a, return_state = True)         
densor = Dense(n_values, activation='softmax')     

def djmodel(Tx, n_a, n_values):
    X = Input(shape=(Tx, n_values))
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0
    
    outputs = []
    
    for t in range(Tx):
        x = Lambda(lambda x: X[:,t,:])(X)
        x = reshapor(x)
        a, _, c = LSTM_cell(x, initial_state=[a, c])
        out = densor(a)
        outputs.append(out)
        
    
    model = Model(inputs=[X, a0, c0], outputs=outputs)
    
    return model

model = djmodel(Tx = 30 , n_a = 64, n_values = 78)
print(model)
opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

m = 60
a0 = np.zeros((m, n_a))
c0 = np.zeros((m, n_a))

model.fit([X, a0, c0], list(Y), epochs=100)


def music_inference_model(LSTM_cell, densor, n_values = 78, n_a = 64, Ty = 100):
    """
    Uses the trained "LSTM_cell" and "densor" from model() to generate a sequence of values.
    """
    # Define the input of model with a shape 
    x0 = Input(shape=(1, n_values))
    
    # Define s0, initial hidden state for the decoder LSTM
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0
    x = x0
    

    outputs = []
    
    # Loop over Ty and generate a value at every time step
    for t in range(Ty):
        
        # Perform one step of LSTM_cell
        a, _, c = LSTM_cell(x, initial_state=[a, c])
        
        # Apply Dense layer to the hidden state output of the LSTM_cell
        out = densor(a)

        # Append the prediction "out" to "outputs". out.shape = (None, 78)
        outputs.append(out)

        x = Lambda(one_hot)(out)
        
    # Create model instance
    inference_model = Model(inputs=[x0, a0, c0], outputs=outputs)
    
    return inference_model

inference_model = music_inference_model(LSTM_cell, densor, n_values = 78, n_a = 64, Ty = 100)

#Generate music
out_stream = generate_music(inference_model)

