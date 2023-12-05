# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 14:24:00 2023

@author: Hovsep Touloujian
"""

import gym
import pdb
from gym import spaces
from gym.envs.toy_text import discrete
import numpy as np
from numpy import matlib as mb
import seeding

import matplotlib.pyplot as plt
from control.matlab import *

from SPM_Params import *

# TRAINING settings
 #settings 
settings={}
settings['sample_time']=30
settings['periodic_test']=10 # number of save points.

settings['number_of_training_episodes']=3000
settings['number_of_episode_steps'] = 30000
settings['number_of_training']= 3 # Number of training.
settings['episodes_number_test']=10 # Number of testing.

#reference for the state of charge
control_settings={}
control_settings['references']={}
control_settings['references']['soc']=0.9; 
control_settings['max_charging_current'] = -3.0*OneC(p)


# constraints    
control_settings['constraints']={}
control_settings['constraints']['temperature']={}
control_settings['constraints']['voltage']={}
control_settings['constraints']['etasLn']={}
control_settings['constraints']['i_s_n'] = {}
control_settings['constraints']['normalized_cssn']={}
control_settings['constraints']['temperature']['max']=35;
control_settings['constraints']['voltage']['max']=4.3;
control_settings['constraints']['i_s_n']['min'] = -2e-5
# control_settings['constraints']['normalized_cssn']['max']=0.932
        
# negative score at which the episode ends
control_settings['max_negative_score']=-1000

class SPM(discrete.DiscreteEnv):

    def __init__(self,sett, cont_sett):
        self.sett = sett
        self.cont_sett = cont_sett

        self.dt = p['dt']

        self.OneC = OneC(p)

        self.ref_SOC = cont_sett['references']['soc']
        self.max_episode_steps = settings['number_of_episode_steps']
        self.episode_step = 0

    def action_space(self):
        return spaces.Box(dtype=np.float32, low = self.cont_sett['max_charging_current'],high = 0, shape = (1,))
    
    def step(self,x_t,a_t):

        is_done = False
        a_t = np.clip(a_t, a_min = self.action_space().low[0], a_max=self.action_space().high[0])[0]

        f = f_SPM(x_t,a_t,p)
        x_tp1 = x_t + self.dt * f[0]

        _, _, _, Tc, _, _, _, _, _, _, _ = state_convert(x_tp1,p)
        V_cell, SOC_p = f[1], f[4]
        i_s = f[-1]
        s_tp1 = np.concatenate((V_cell, SOC_p[0], Tc, i_s))
		
        r_temp = -5*abs(Tc - self.cont_sett['constraints']['temperature']['max']) if Tc > self.cont_sett['constraints']['temperature']['max'] else 0
        r_volt = -10*abs(V_cell-self.cont_sett['constraints']['voltage']['max']) if V_cell > self.cont_sett['constraints']['voltage']['max'] else 0
        r_i_s = -10*abs(i_s-self.cont_sett['constraints']['i_s_n']['min']) if i_s < self.cont_sett['constraints']['i_s_n']['min'] else 0
        r_step = -0.01
        reward = r_temp + r_i_s + r_volt + r_step

        self.episode_step+=1

        if SOC_p[0][0] >= self.ref_SOC or self.episode_step >= self.max_episode_steps:
            is_done = True
        else:
            is_done = False
        
        # print(SOC_p[0][0],self.episode_step, is_done)
        return x_tp1, s_tp1, reward, is_done
    
    def reset(self):
        self.episode_step = 0
        return set_sample(self.cont_sett['constraints']['i_s_n']['min'],0,0.3,0)
    

    
