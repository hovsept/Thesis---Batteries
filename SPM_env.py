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
from settings import *

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
		
        r_temp = -0.5*abs(Tc - self.cont_sett['constraints']['temperature']['max']) if Tc > self.cont_sett['constraints']['temperature']['max'] else 0
        r_volt = -0.05*abs(V_cell-self.cont_sett['constraints']['voltage']['max']) if V_cell > self.cont_sett['constraints']['voltage']['max'] else 0
        r_i_s = -1000*abs(i_s) 
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
        return sample_IC(p, SOC_min = 0., SOC_max = 0.2)
    

    
