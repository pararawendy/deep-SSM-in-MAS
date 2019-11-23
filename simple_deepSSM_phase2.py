# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 21:00:49 2019

@author: parar
"""

#prepare the environment
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
tfe = tf.contrib.eager #needed in training phase
tf.enable_eager_execution() #activate tensorflow eager execution

import tensorflow_probability as tfp
tfd = tfp.distributions #handy to work with distributions

import numpy as np

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.python.ops.rnn import _transpose_batch_time 

#load the train data
mastertoy = np.genfromtxt('C:\\deep_SSM\\TRAINDATW.csv', delimiter=',')
mastertoy = np.array(mastertoy, dtype='float32')
mastertoy.shape

'''
column of mastertoy
6 col L-R: 
1. action speed, 
2. action direction, 
3. coord.x, 
4. coord.y, 
5. actual speed, 
6. actual direction. 

row = 20K, new episode begin every row = 20k+1
'''
act_direct = mastertoy[:,1] #the only needed from mastertoy: second action, i.e. heading
act_direct[act_direct < 0] = 0 
act_direct[act_direct > 0] = 1
act_direct = act_direct.reshape((act_direct.shape[0],1))
act_direct.shape #to make it (30000,1)

#call true_latent data (from Phase 1 nodel with rank 2)
masterlatent = np.genfromtxt('C:\\deep_SSM\\envstate_rank2.csv', delimiter=',')
masterlatent = np.array(masterlatent, dtype='float32')
masterlatent.shape

#Global variable
pre_seq_length = 20 #original trajectory length of train data
seq_length = 19 #effective length used for modelling phase 2 is 20-1 (the first opponent action is assummed given)
num_seq = 1500 #number of sequences/trajectories in training data

#preparing the shape of training data
temp_action = tf.reshape(act_direct, shape=(num_seq, pre_seq_length, -1))
temp_action = _transpose_batch_time(temp_action)
action = temp_action[-seq_length:,:,:] 

temp_latent = tf.reshape(masterlatent, shape=(num_seq, pre_seq_length, -1))
temp_latent = _transpose_batch_time(temp_latent)
latent = temp_latent[:seq_length,:,:]

train_data = tf.concat([action, latent], -1)
train_data.shape

#define a class of the model (simple: rank 0), in fact it is also an RNN cell
class SSM_phase2(tf.keras.Model):
    def __init__(self, latent_dim = 2, emission_dim = 1, phase1latent_dim = 4):
        super(SSM_phase2, self).__init__()
        self.latent_dim = latent_dim
        self.emission_dim = emission_dim #opponent action dimension tobe predicted
        self.phase1latent_dim = phase1latent_dim
        self.input_dim = (self.latent_dim + self.emission_dim + self.phase1latent_dim) + (self.latent_dim + self.phase1latent_dim) # inference net and transition net     
        self.output_dim = (2 * self.latent_dim) + (2 * self.latent_dim) + (self.emission_dim) #output RNNcell define distributions: mean&logvar of infer, trans & only mean for generative
        self.seq_length = seq_length #from global variable seq_length
        
        #inference net
        inference_input = Input(shape=(self.latent_dim + self.emission_dim + self.phase1latent_dim,), name='inference_input')
        layer_1 = Dense(15, activation='tanh')(inference_input)
        layer_2 = Dense(15, activation='tanh')(layer_1)
        infer_mean_logvar = Dense(2 * self.latent_dim)(layer_2)
        self.inference_net = Model(inference_input, infer_mean_logvar, name='inference_net')
        
        #transition net
        trans_input = Input(shape=(self.latent_dim + self.phase1latent_dim,), name='transition_net')
        layer_1a = Dense(15, activation='tanh')(trans_input)
        layer_2a = Dense(15, activation='tanh')(layer_1a)
        trans_mean_logvar = Dense(2 * self.latent_dim)(layer_2a)
        self.transition_net = Model(trans_input, trans_mean_logvar, name='transition_net')
        
        #emission net
        latent_inputs = Input(shape=(self.latent_dim,), name='s_sampling')
        layer_3 = Dense(15, activation='relu')(latent_inputs)
        layer_4 = Dense(15, activation='relu')(layer_3)
        obs_mean = Dense(self.emission_dim)(layer_4)
        self.generative_net = Model(latent_inputs, obs_mean, name='emission_net') #only the mean is learned (fixed variance)
    
    def encode(self, input_infer):
        infer_mean, infer_logvar = tf.split(self.inference_net(input_infer), num_or_size_splits=2, axis=-1)
        return infer_mean, infer_logvar
    
    def reparameterize(self, mean, logvar): #sample latent variable from standard normal (reparameterization trick)
        eps = tf.random_normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean
    
    def decode(self, z):
        return self.generative_net(z)
    
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
        infer_mean, infer_logvar = self.encode(inputs[:,:(self.latent_dim + self.emission_dim + self.phase1latent_dim)])
        next_state = self.reparameterize(infer_mean, infer_logvar) #the sampled latent variable
        trans_mean, trans_logvar = self.trans(inputs[:,(self.latent_dim + self.emission_dim + self.phase1latent_dim):])
        action_logit = self.decode(next_state) #logit of bernoulli
        output = tf.concat([infer_mean, infer_logvar, trans_mean, trans_logvar, action_logit], -1)
        return output, next_state


#define a function for prediction, utilize tf.nn.raw_rnn API
def SSM_model(model, inputs):
    #variables needed for loop_fn
    sequence_length = tf.shape(inputs).numpy()[0]
    output_dim = model.output_dim #output dimension of the model/RNN cell
    output_ta = tf.TensorArray(dtype=tf.float32, size=sequence_length) #for saving state
    inputs_ta = tf.TensorArray(dtype=tf.float32, size=sequence_length, clear_after_read=False)
    inputs_ta = inputs_ta.unstack(inputs) #convert to tensor array

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
            next_input = tf.concat([inputs_ta.read(time), next_cell_state, inputs_ta.read(time)[:,1:], next_cell_state], -1)
        
        return (elements_finished, next_input, next_cell_state, emit_output, next_loop_state) 
    
    #output yielded here using tf.nn.raw_rnn
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
    action_logit = outputs[:,:,-1] #logit of Bernoulli distribution 
    
    #the true observation we want to recover
    target = inputs[:,:,0] #recall training data format for phase 2
    
    #transform logvar to std
    infer_std = tf.sqrt(tf.exp(infer_logvar))
    trans_std = tf.sqrt(tf.exp(trans_logvar))
    

    #distribution of each module
    infer_dist = tfd.MultivariateNormalDiag(infer_mean,infer_std)
    trans_dist = tfd.MultivariateNormalDiag(trans_mean,trans_std)
    
        
    #log likelihood of observation
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=action_logit, labels=target)
        
    #KL of (q|p)
    kl = tfd.kl_divergence(infer_dist, trans_dist) #shape = batch_size
        
    #the loss
    loss = cross_ent + kl
    overall_loss = tf.cast(tf.reduce_sum(loss), dtype='float32')
    
    return overall_loss

#computing gradient function
def compute_gradients(model, inputs):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs)
  return tape.gradient(loss_value, model.trainable_variables), loss_value


#train the model
model = SSM_phase2()
train_loss_results = []
global_step = tf.Variable(0)

learning_rate = 0.005
optimizer = tf.train.AdamOptimizer(learning_rate)

for epoch in np.arange(0,150): #train for 150 epochs
    epoch_loss_avg = tfe.metrics.Mean()
    
    grads, loss_value = compute_gradients(model, train_data)
    optimizer.apply_gradients(zip(grads, model.trainable_variables), global_step = global_step)
        
    #track progress
    epoch_loss_avg(loss_value) #add current batch loss
    #training loop (batch_size = batch_size)
    
    #end epoch
    train_loss_results.append(epoch_loss_avg.result())
    
    if epoch % 5 == 0:
        print('REP 1, Epoch: {},      Loss: {}'.format(epoch, train_loss_results[epoch]))
        
#save model weights
model.save_weights('C:\\deep_SSM\\model_phase2.h5')



