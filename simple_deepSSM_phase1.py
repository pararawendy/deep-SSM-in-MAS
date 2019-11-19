# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 20:35:56 2019

@author: parar
"""


#prepare the environment
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
tfe = tf.contrib.eager #needed in training phase
tf.enable_eager_execution() #tensorflow eager execution

import tensorflow_probability as tfp
tfd = tfp.distributions #handy to manipulate distributions

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.python.ops.rnn import _transpose_batch_time 

#load the train data
mastertoy = np.genfromtxt('C:\\deep_SSM\\TRAINDATW.csv', delimiter=',')
mastertoy = np.array(mastertoy, dtype='float32')
mastertoy.shape

'''
RECALL column of mastertoy
6 columns L-R: 
0. action speed, 
1. action direction, 
2. coord.x, 
3. coord.y, 
4. actual speed, 
5. actual direction. 
'''

#prepare the data
act_speed = mastertoy[:,0] #the first action: speed
act_speed[act_speed < 1] = 0 #make the data binary {0,1}

act_direct = mastertoy[:,1] #second action: heading
act_direct[act_direct < 0] = 0 
act_direct[act_direct > 0] = 1

obs_x = mastertoy[:,2] #x-axis coordinate
obs_y = mastertoy[:,3] #y-axis coordinate

#Global variable
seq_length = 20 #length of each trajectory
num_seq = 1500 #number of sequences/trajectories in training data

#reshape the inputs as 3D tensor (seq_length,num_seq,value)
act_speed = act_speed.reshape((-1, seq_length)).T
act_direct = act_direct.reshape((-1, seq_length)).T
obs_x = obs_x.reshape((-1, seq_length)).T
obs_y = obs_y.reshape((-1, seq_length)).T
train_data = tf.stack([act_speed, act_direct, obs_x, obs_y], axis=-1)
train_data.shape # i = time i, j = sample j, k = action k : (20, 1500, 4)

#define a class of the model (basic, rank 0), in fact it is also an RNN cell
class SSM(tf.keras.Model):
    def __init__(self, latent_dim = 4, output_obs_dim = 2, action_dim = 2):
        super(SSM, self).__init__()
        self.latent_dim = latent_dim
        self.output_obs_dim = output_obs_dim
        self.action_dim = action_dim
        self.input_dim = (self.latent_dim + self.output_obs_dim + self.action_dim) + (self.latent_dim + self.action_dim) # inference net and transition net     
        self.output_dim = (2 * self.latent_dim) + (2 * self.latent_dim) + (self.output_obs_dim) #output RNNcell define distributions: mean&logvar of infer, trans & only mean for generative
        self.seq_length = seq_length #from global variable seq_length
        
        #inference net
        inference_input = Input(shape=(self.latent_dim + self.output_obs_dim + self.action_dim,), name='inference_input')
        layer_1 = Dense(15, activation='tanh')(inference_input)
        layer_2 = Dense(15, activation='tanh')(layer_1)
        infer_mean_logvar = Dense(2 * self.latent_dim)(layer_2)
        self.inference_net = Model(inference_input, infer_mean_logvar, name='inference_net')
        
        #transition net
        trans_input = Input(shape=(self.latent_dim + self.action_dim,), name='transition_net')
        layer_1a = Dense(15, activation='tanh')(trans_input)
        layer_2a = Dense(15, activation='tanh')(layer_1a)
        trans_mean_logvar = Dense(2 * self.latent_dim)(layer_2a)
        self.transition_net = Model(trans_input, trans_mean_logvar, name='transition_net')
        
        #emission net
        latent_inputs = Input(shape=(self.latent_dim,), name='s_sampling')
        layer_3 = Dense(15, activation='relu')(latent_inputs)
        layer_4 = Dense(15, activation='relu')(layer_3)
        obs_mean = Dense(self.output_obs_dim)(layer_4)
        self.generative_net = Model(latent_inputs, obs_mean, name='emission_net') #only the mean is learned (fixed variance)
    
    def encode(self, input_infer):
        infer_mean, infer_logvar = tf.split(self.inference_net(input_infer), num_or_size_splits=2, axis=-1)
        return infer_mean, infer_logvar
    
    def reparameterize(self, mean, logvar): #sample latent variable from standard normal (reparameterization trick)
        eps = tf.random_normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean
    
    def decode(self, s):
        return self.generative_net(s)
    
    def trans(self,input_trans):
        trans_mean, trans_logvar = tf.split(self.transition_net(input_trans), num_or_size_splits=2, axis=-1)
        return trans_mean, trans_logvar
    
    @property
    def state_size(self): #recurrent cell state dimension 
        return self.latent_dim
    
    @property
    def output_size(self): #recurrent cell output dimension 
        return self.output_dim #concatenate all outputs from 3 networks
    
    @property
    def zero_state(self): #recurrent cell initial state 
        return tf.zeros([num_seq, self.latent_dim]) #depends on global variable num_seq
    
    def __call__(self, inputs, state): #behavior/logic of the cell
        infer_mean, infer_logvar = self.encode(inputs[:,:(self.latent_dim + self.output_obs_dim + self.action_dim)])
        next_state = self.reparameterize(infer_mean, infer_logvar) #the sampled latent variable
        trans_mean, trans_logvar = self.trans(inputs[:,(self.latent_dim + self.output_obs_dim + self.action_dim):])
        obs_mean = self.decode(next_state)
        output = tf.concat([infer_mean, infer_logvar, trans_mean, trans_logvar, obs_mean], -1)
        return output, next_state
    
#define a function for prediction, utilize tf.nn.raw_rnn API
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
    
    #output obtained here using tf.nn.raw_rnn API
    outputs_ta, _, latent_ta = tf.nn.raw_rnn(model, loop_fn)
    outputs = outputs_ta.stack() #stack to make it tensor (from tensor array)
    outputs = tf.where(tf.is_nan(outputs), tf.zeros_like(outputs), outputs) #convert nan entries to zero

    latent = latent_ta.stack()
    latent = tf.where(tf.is_nan(latent), tf.zeros_like(latent), latent)
    
    return [outputs, latent]

#the loss function (negative ELBO)
def loss(model, inputs):
    outputs, _ = SSM_model(model, inputs) #only RNN output is needed for computing loss
    
    #allocate the corresponding output component
    infer_mean = outputs[:,:,:model.latent_dim]  #mean of latent variable from  inference net
    infer_logvar = outputs[:,:,model.latent_dim : (2 * model.latent_dim)]
    trans_mean = outputs[:,:,(2 * model.latent_dim):(3 * model.latent_dim)] #mean of latent variable from transition net
    trans_logvar = outputs[:,:, (3 * model.latent_dim):(4 * model.latent_dim)]
    obs_mean = outputs[:,:,(4 * model.latent_dim):] #mean of observation from  generative net
    
    #the true observation we want to recover
    target = inputs[:,:,2:4] #recall training data format
    
    #transform logvar to std
    infer_std = tf.sqrt(tf.exp(infer_logvar))
    trans_std = tf.sqrt(tf.exp(trans_logvar))
    obs_std = tf.constant(1.,shape=(tf.shape(obs_mean).numpy())) #fixed predicted observation std = 1
    
    #distribution of each module
    infer_dist = tfd.MultivariateNormalDiag(infer_mean,infer_std)
    trans_dist = tfd.MultivariateNormalDiag(trans_mean,trans_std)
    obs_dist = tfd.MultivariateNormalDiag(obs_mean,obs_std)
        
    #log likelihood of observation
    likelihood = obs_dist.prob(target) 
    likelihood = tf.clip_by_value(likelihood, 1e-37, 1)
    log_likelihood = tf.log(likelihood)
        
    #KL of (q|p)
    kl = tfd.kl_divergence(infer_dist, trans_dist) #analytic KL divergence for two Gaussians
        
    #the loss
    loss = - log_likelihood + kl #negative of ELBO
    overall_loss = tf.cast(tf.reduce_sum(loss), dtype='float32')
    
    return overall_loss


#Function to compute gradients
def compute_gradients(model, inputs):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs)
  return tape.gradient(loss_value, model.trainable_variables), loss_value

#train the model
model = SSM() #model instantiation
train_loss_results = []
global_step = tf.Variable(0)

#train the model
learning_rate = 0.005
optimizer = tf.train.AdamOptimizer(learning_rate)

for epoch in np.arange(0,200): #train for 200 epochs
    epoch_loss_avg = tfe.metrics.Mean()
    
    grads, loss_value = compute_gradients(model, train_data)
    optimizer.apply_gradients(zip(grads, model.trainable_variables), global_step = global_step)
        
    #track progress
    epoch_loss_avg(loss_value) #add current batch loss
    
    #end epoch
    train_loss_results.append(epoch_loss_avg.result())
    
    if epoch % 5 == 0:
        print('Epoch: {},      Loss: {}'.format(epoch, train_loss_results[epoch]))

#save model weights
model.save_weights('C:\\deep_SSM\\model_rank0.h5')


'''
creating latent variables used in the second phase
'''

#producing latent variables as new features for the next phase (environment state)
rep = 100 #number of running replication, to reduce variance
latents_to_average = np.zeros(shape=(rep,seq_length,num_seq,model.latent_dim))

for i in range(rep):
    latents_to_average[i] = SSM_model(model, train_data)[1]
    
latents_to_average = tf.convert_to_tensor(latents_to_average)
env_state = tf.reduce_mean(latents_to_average, axis = 0)
env_state = _transpose_batch_time(env_state) #adjust with common format
env_state = tf.reshape(env_state, (-1, model.latent_dim)) #make it 2D

#saving the results
np.savetxt('C:\\deep_SSM\\envstate_rank0.csv',env_state,delimiter = ',')
