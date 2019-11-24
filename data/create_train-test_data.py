# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 16:24:54 2019

@author: parar
"""

''' Generating the data:
    Data of the toy problem, i.e. 2D sailing boat trajectories
    There are two kinds of data: 
        1. deterministic (when we do not consider noise in the boat dynamics)
        2. stochastic (when we do consider the noise)
        
    This is the code used to generate training data, which has 1500 trajectories each with 20 time steps long
    Note that it is easy to generate testing data, by simply change global configuration variables
'''

#prepare the environment
import numpy as np
import math

#global configuration variables
seq_length = 20 #length of trajectories created
num_seq = 1500 #number of trajectories created

'''
DETERMINISTIC SYSTEM DYNAMICS
'''

#initiate container
act_speed = np.zeros((num_seq,seq_length), dtype=float)
act_direct = np.zeros((num_seq,seq_length), dtype=float)
coord_x = np.zeros((num_seq,seq_length), dtype=float)
coord_y = np.zeros((num_seq,seq_length), dtype=float)
actual_speed = np.zeros((num_seq,seq_length), dtype=float)
actual_direct = np.zeros((num_seq,seq_length), dtype=float)

#generate data recursively
for eps in range(num_seq):
    for t in range(seq_length):
        rand_1 = np.random.rand()
        if rand_1 <= 0.5:
            act_speed[eps,t] = 1
        else : act_speed[eps,t] = -1
        
        if t == 0:
            rand_2 = np.random.rand()
            if rand_2 <= 0.5:
                act_direct[eps,t] = math.pi/4
            else : act_direct[eps,t] = -math.pi/4
            
            actual_speed[eps,t] = act_speed[eps,t]
            actual_direct[eps,t] = act_direct[eps,t]
            
            coord_x[eps,t] = actual_speed[eps,t] * math.cos(actual_direct[eps,t])
            coord_y[eps,t] = actual_speed[eps,t] * math.sin(actual_direct[eps,t])
            
        else :
            if np.abs(coord_x[eps,t-1]) >= np.abs(coord_y[eps,t-1]):
                   act_direct[eps,t] = math.pi/4
            else : act_direct[eps,t] = -math.pi/4
               
            actual_speed[eps,t] = actual_speed[eps,t-1] + act_speed[eps,t]
            actual_direct[eps,t] = actual_direct[eps,t-1] + act_direct[eps,t]
           
            coord_x[eps,t] = coord_x[eps,t-1] + actual_speed[eps,t] * math.cos(actual_direct[eps,t])
            coord_y[eps,t] = coord_y[eps,t-1] + actual_speed[eps,t] * math.sin(actual_direct[eps,t])


'''
STOCHASTIC SYSTEM DYNAMICS

Noise implemented:
    action speed : Normal(0,0.1)
    action direct : Normal(0,pi/60)
    coordinat x : Normal(0,0.1)
    coordinaty : Normal(0,0.1)
'''

#initiate container (postfix _w stands for 'with_noise')
act_speedw = np.zeros((num_seq,seq_length), dtype=float)
act_directw = np.zeros((num_seq,seq_length), dtype=float)
delta_speed = np.zeros((num_seq,seq_length), dtype=float)
delta_direct = np.zeros((num_seq,seq_length), dtype=float)
coord_xw = np.zeros((num_seq,seq_length), dtype=float)
coord_yw = np.zeros((num_seq,seq_length), dtype=float)
actual_speedw = np.zeros((num_seq,seq_length), dtype=float)
actual_directw = np.zeros((num_seq,seq_length), dtype=float)

#generate data recursively
for eps in range(num_seq):
    for t in range(seq_length):
        act_speedw[eps,t] = act_speed[eps,t]
        delta_speed[eps,t] = act_speedw[eps,t] + np.random.normal(0,0.1)
        
        if t == 0:
            act_directw[eps,t] = act_direct[eps,t]
            delta_direct[eps,t] = act_directw[eps,t] + np.random.normal(0,math.pi/60)
            
            actual_speedw[eps,t] = delta_speed[eps,t]
            actual_directw[eps,t] = delta_direct[eps,t]
            
            coord_xw[eps,t] = actual_speedw[eps,t] * math.cos(actual_directw[eps,t]) + np.random.normal(0,0.1)
            coord_yw[eps,t] = actual_speedw[eps,t] * math.sin(actual_directw[eps,t]) + np.random.normal(0,0.1)
            
        else :
            if np.abs(coord_xw[eps,t-1]) >= np.abs(coord_yw[eps,t-1]):
                   act_directw[eps,t] = math.pi/4
            else : act_directw[eps,t] = -math.pi/4
           
            delta_direct[eps,t] = act_directw[eps,t] + np.random.normal(0,math.pi/60)
           
            actual_speedw[eps,t] = actual_speedw[eps,t-1] + delta_speed[eps,t]
            actual_directw[eps,t] = actual_directw[eps,t-1] + delta_direct[eps,t]
           
            coord_xw[eps,t] = coord_xw[eps,t-1] + actual_speedw[eps,t] * math.cos(actual_directw[eps,t]) + np.random.normal(0,0.1)
            coord_yw[eps,t] = coord_yw[eps,t-1] + actual_speedw[eps,t] * math.sin(actual_directw[eps,t]) + np.random.normal(0,0.1)
      
'''SAVING THE DATA'''

#deterministic system dynamics, all flatten to make it 1D 
act_speed = act_speed.flatten()
act_direct = act_direct.flatten()         
actual_speed = actual_speed.flatten()               
actual_direct = actual_direct.flatten()       
coord_x = coord_x.flatten()
coord_y = coord_y.flatten()

# stack together to make it 2D (to be saved in csv)
master = np.vstack((act_speed,act_direct,coord_x,coord_y,actual_speed,actual_direct)).T
master.shape #check the shape

np.savetxt('C:\\deep_SSM\\TRAINDAT.csv',master,delimiter = ',')

#stochastic system dynamics    
act_speedw = act_speedw.flatten()
act_directw = act_directw.flatten()      
actual_speedw = actual_speedw.flatten()
actual_directw = actual_directw.flatten()
coord_xw = coord_xw.flatten()
coord_yw = coord_yw.flatten()

masterw = np.vstack((act_speedw,act_directw,coord_xw,coord_yw,actual_speedw,actual_directw)).T

np.savetxt('C:\\deep_SSM\\TRAINDATW.csv',masterw,delimiter = ',')
