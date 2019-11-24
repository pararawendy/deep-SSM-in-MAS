# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 22:41:52 2019

@author: parar
"""

'''
In this code we give the implementation of using model Phase1-
to predict observation (x,y) coordinate

Note we take model with covariance matrix rank of 2 here.

Implementation for the other ranks is similar.
'''

#prepare the environment
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
tfe = tf.contrib.eager 
tf.enable_eager_execution() #tensorflow eager execution

import tensorflow_probability as tfp
tfd = tfp.distributions #handy to manipulate distributions

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model 

#defining class of the model with general covariance matrix, in fact it is also RNN cell
class SSM(tf.keras.Model):
    def __init__(self, rank_F, latent_dim = 4, action_dim = 2, output_obs_dim = 2):
        super(SSM, self).__init__()
        self.latent_dim = latent_dim
        self.output_obs_dim = output_obs_dim
        self.action_dim = action_dim
        self.rank_F = rank_F #covariance matrix rank
        self.input_dim = (self.latent_dim + self.output_obs_dim + self.action_dim) + (self.latent_dim + self.action_dim) # inference net and transition net     
        self.output_dim = ((2 + self.rank_F) * self.latent_dim) + ((2 + self.rank_F) * self.latent_dim) + (self.output_obs_dim) #output RNNcell: mean&logvar of infer, trans & only mean for generative
        
        #inference net
        inference_input = Input(shape=(self.latent_dim + self.output_obs_dim + self.action_dim,), name='inference_input')
        layer_1 = Dense(15, activation='tanh')(inference_input)
        layer_2 = Dense(15, activation='tanh')(layer_1)
        infer_output = Dense((2 + self.rank_F) * self.latent_dim)(layer_2)
        self.inference_net = Model(inference_input, infer_output, name='inference_net')
        
        #transition net
        trans_input = Input(shape=(self.latent_dim + self.action_dim,), name='transition_net')
        layer_1a = Dense(15, activation='tanh')(trans_input)
        layer_2a = Dense(15, activation='tanh')(layer_1a)
        trans_output = Dense((2 + self.rank_F) * self.latent_dim)(layer_2a)
        self.transition_net = Model(trans_input, trans_output, name='transition_net')
        
        #generative net
        latent_inputs = Input(shape=(self.latent_dim,), name='s_sampling')
        layer_3 = Dense(15, activation='relu')(latent_inputs)
        layer_4 = Dense(15, activation='relu')(layer_3)
        obs_mean = Dense(self.output_obs_dim)(layer_4)
        self.generative_net = Model(latent_inputs, obs_mean, name='generative_net')
    
    def encode(self, input_infer):
        infer_output = self.inference_net(input_infer)
        infer_mean, infer_logvar = tf.split(infer_output[:,: 2*self.latent_dim], num_or_size_splits=2, axis=-1)
        infer_cov = infer_output[:, 2*self.latent_dim:]
        return infer_mean, infer_logvar, infer_cov #now also returning covariance
    
    def reparameterize(self, mean, logvar, cov): #sample latent variable from standard normal (reparameterization trick)       
        eps_hi = tf.random_normal(shape=(num_seq, self.latent_dim))
        eps_lo = tf.random_normal(shape=(num_seq, self.rank_F, 1))
        cov_factor = tf.reshape(cov, shape=(num_seq, self.latent_dim, self.rank_F))
        cov_term = tf.reshape(tf.matmul(cov_factor,eps_lo),shape=mean.shape)
        return  cov_term +  eps_hi * tf.exp(logvar * .5) + mean
    
    def decode(self, s):
        return self.generative_net(s)
    
    def trans(self,input_trans):
        trans_output = self.transition_net(input_trans)
        trans_mean, trans_logvar = tf.split(trans_output[:,:2*self.latent_dim], num_or_size_splits=2, axis=-1)
        trans_cov = trans_output[:, 2*self.latent_dim:]
        return trans_mean, trans_logvar, trans_cov #now also returning covariance
    
    @property
    def state_size(self): #recurrent cell state dimension 
        return self.latent_dim
    
    @property
    def output_size(self): #recurrent cell output dimension 
        return self.output_dim
    
    @property
    def zero_state(self): #recurrent cell initial state 
        return tf.zeros([num_seq, self.latent_dim]) #depends on global variable num_seq #global variable init_state
    
    def __call__(self, inputs, state): #behavior/logic of the cell
        infer_mean, infer_logvar, infer_cov = self.encode(inputs[:,:(self.latent_dim + self.output_obs_dim + self.action_dim)])
        next_state = self.reparameterize(infer_mean, infer_logvar, infer_cov) #the sampled latent variable
        trans_mean, trans_logvar, trans_cov = self.trans(inputs[:,(self.latent_dim + self.output_obs_dim + self.action_dim):])
        obs_mean = self.decode(next_state)
        output = tf.concat([infer_mean, infer_logvar, infer_cov, trans_mean, trans_logvar, trans_cov, obs_mean], -1)
        return output, next_state

#call the model
model = SSM(rank_F = 2)

#load the weights
model.load_weights('C:\\deep_SSM\\model_rank2.h5')

#function to operate the cell
def SSM_model(model, inputs):
    #variables needed for loop_fn
    sequence_length = tf.shape(inputs).numpy()[0]
    output_dim = model.output_dim #output dimension of the model/RNN cell
    output_ta = tf.TensorArray(dtype=tf.float32, size=sequence_length) #for saving state
    inputs_ta = tf.TensorArray(dtype=tf.float32, size=sequence_length, clear_after_read=False)
    inputs_ta = inputs_ta.unstack(inputs)

    #define loop_fn, the input-output data flow of the cell
    def loop_fn(time, cell_output, cell_state, loop_state):
        emit_output = cell_output 
        if cell_output is None: # when time == 0
            next_cell_state = tf.zeros([num_seq, model.latent_dim])
            emit_output = tf.zeros([output_dim])
            next_loop_state = output_ta
            
        else : # when time > 0
            next_cell_state = cell_state
            next_loop_state = loop_state.write(time-1, next_cell_state) #s_1 is put at index 0
            
        elements_finished = (time >= sequence_length)
        finished = tf.reduce_all(elements_finished)
        
        if finished :
            next_input = tf.zeros(shape=(output_dim), dtype=tf.float32)
        else :
            next_input = tf.concat([inputs_ta.read(time), next_cell_state, inputs_ta.read(time)[:,:2], next_cell_state], -1)
        
        return (elements_finished, next_input, next_cell_state, emit_output, next_loop_state) 
    
    #output yielded here using tf.nn.raw_rnn
    outputs_ta, _, latent_ta = tf.nn.raw_rnn(model, loop_fn)
    outputs = outputs_ta.stack() #stack to make it tensor (from tensor array)
    outputs = tf.where(tf.is_nan(outputs), tf.zeros_like(outputs), outputs) #convert nan entries to zero

    latent = latent_ta.stack()
    latent = tf.where(tf.is_nan(latent), tf.zeros_like(latent), latent)
    
    return [outputs, latent]

#function to predict the observations (x,y) coordinate
def SSM_predict(model, inputs):
    predicted_obs = []
    #predict first observation (t=0), note: t index
    trans_input = tf.concat([inputs[0,:,0:2],tf.zeros([num_seq, model.latent_dim])],-1)
    mean, _, _ = model.trans(trans_input)
    obs = model.generative_net(mean)
    predicted_obs.append(obs)
    for t in np.arange(1,seq_length):
        trans_input = tf.concat([inputs[t,:,0:2],empirical_latent[t]],-1)
        mean, _, _ = model.trans(trans_input)
        obs = model.generative_net(mean)
        predicted_obs.append(obs)
    return tf.convert_to_tensor(predicted_obs)

#load test data
pre_testdat = np.genfromtxt('C:\\deep_SSM\\TESTDATW.csv', delimiter=',')
pre_testdat = np.array(pre_testdat, dtype='float32')

#prepare the data
act_speed = pre_testdat[:,0] #the first action: speed
act_speed[act_speed < 1] = 0 #make the data binary {0,1}

act_direct = pre_testdat[:,1] #second action: heading
act_direct[act_direct < 0] = 0 
act_direct[act_direct > 0] = 1

obs_x = pre_testdat[:,2] #x-axis coordinate
obs_y = pre_testdat[:,3] #y-axis coordinate

#necessary global variables
seq_length = 20 #length of each trajectory
num_seq = 200 #number of sequences/trajectories in test data

#reshape the inputs as 3D tensor (seq_length,num_seq,value)
act_speed = act_speed.reshape((-1, seq_length)).T
act_direct = act_direct.reshape((-1, seq_length)).T
obs_x = obs_x.reshape((-1, seq_length)).T
obs_y = obs_y.reshape((-1, seq_length)).T
test_data = tf.stack([act_speed, act_direct, obs_x, obs_y], axis=-1)
test_data.shape


#prepare latent variables
rep = 100 #number of running replication, to reduce variance
latents_to_average = np.zeros(shape=(rep,seq_length,num_seq,model.latent_dim))

for i in range(rep):
    latents_to_average[i] = SSM_model(model, test_data)[1]
    
latents_to_average = tf.convert_to_tensor(latents_to_average)
empirical_latent = tf.reduce_mean(latents_to_average, axis = 0)
empirical_latent = tf.cast(empirical_latent, dtype='float32')

#let's predict!
prediction = SSM_predict(model, test_data)

chosen_eps = np.random.choice(num_seq, size=1) #index of sample episode chosen
predicted_obs2 = prediction[:,chosen_eps[0],:] #model prediction for the chosen episode

true_obs = test_data[:,chosen_eps[0],2:4] #ground truth observation for the chosen episode

#transform to array
prediction_x =predicted_obs2[:,0].numpy()
prediction_y =predicted_obs2[:,1].numpy()

true_x = true_obs[:,0].numpy()
true_y = true_obs[:,1].numpy()

#the plot
fig = plt.figure(figsize=(10,12))
ax = fig.add_subplot(211)
ax.plot(true_x, true_y, 'k-o',label='True')
ax.plot(prediction_x, prediction_y, 'b--o',label='Prediction')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
#ax.set_title('Boat Sample Trajectory')
ax.legend()
ax.grid()
