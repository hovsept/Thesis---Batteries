# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 16:22:09 2023

@author: Hovsep Touloujian
"""

import numpy as np
from SPM_Params import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.optimize
import torch
from functools import *

# import sys
# sys.path.append('RL-Batt-DDPG-main\OutputFeedback_RL')

from ddpg_agent import Agent

# x0 = sample_IC(p)
x0 = np.array([2927,2927,2927,25611,25611,25611,25,25,5e-9,13,4800,4800,4800]).reshape((13,1))
# x0 = set_sample(2.5e-5,0,0.3,0)
cs_n0, cs_p0, Ts0, Tc0, L_sei0, Q0, c_solv0, c_surf_solv0, cs_surf_n0, cs_surf_p0, T_amb = state_convert(x0,p)

def SPM(x0,p,input_mode = 2):
     #0: open loop, 1: feedback, 2: CC-CV, 3: DDPG Actor Output Feedback

    out = {}
    out['t'] = np.arange(0,p['Tf'],p['dt'])

    out['I_cell'] = np.zeros((len(out['t'])))
    if input_mode == 0:
        out['I_cell'] = -OneC(p)*np.ones((len(out['t'],)))
    elif input_mode==1:
        out['V_ref'] = 4.1
        out['SOC_ref'] = 0.9
        out['I_cell'][0] = 0
    elif input_mode==2:
        out['V_max'] = 4.1
        out['I_max'] = 1.5*OneC(p)
        out['I_min'] = 0.1
    elif input_mode == 3:
        i_training = 0
        agent = Agent(state_size=3, action_size=1, random_seed=i_training)  # the number of state is 496.

        start_episode = 80
        agent.actor_local.load_state_dict(torch.load('results/SPM_training_results/training'+str(i_training)+'/episode'+str(start_episode)+'/checkpoint_actor_'+str(start_episode)+'.pth',map_location=torch.device('cpu')))
        agent.actor_optimizer.load_state_dict(torch.load('results/SPM_training_results/training'+str(i_training)+'/episode'+str(start_episode)+'/checkpoint_actor_optimizer_'+str(start_episode)+'.pth',map_location=torch.device('cpu')))
        agent.critic_local.load_state_dict(torch.load('results/SPM_training_results/training'+str(i_training)+'/episode'+str(start_episode)+'/checkpoint_critic_'+str(start_episode)+'.pth',map_location=torch.device('cpu')))
        agent.critic_optimizer.load_state_dict(torch.load('results/SPM_training_results/training'+str(i_training)+'/episode'+str(start_episode)+'/checkpoint_critic_optimizer_'+str(start_episode)+'.pth',map_location=torch.device('cpu')))

        def normalize_outputs(soc, voltage, temperature):
            norm_soc = soc - 0.5
            norm_voltage = (voltage - 3.5) / 1.2
            norm_temperature = (temperature - 298 - 10) / (320 - 298)
            norm_output = np.array([norm_soc, norm_voltage, norm_temperature])
            return norm_output

        min_OUTPUT_value = -3*OneC(p)
        max_OUTPUT_value = 0
    # compute actual action from normalized action 
        def denormalize_input(input_value, min_OUTPUT_value, max_OUTPUT_value):
            output_value=(1+input_value)*(max_OUTPUT_value-min_OUTPUT_value)/2+min_OUTPUT_value
            return output_value

        
        
    n = p['nrn']+p['nrp']+p['n_sei']+4
    out['x'] = np.zeros((n,len(out['t'])))
    out['x'][:,0] = x0.reshape((n,))
    out['V_cell'] = np.zeros((len(out['t'])-1,))
    out['V_oc'] = np.zeros((len(out['t'])-1,))
    out['SOC_n'] = np.zeros((len(out['t'])-1,))
    out['SOC_p'] = np.zeros((len(out['t'])-1,))
    out['i_s'] = np.zeros((len(out['t'])-1,))
    out['is_CV'] = np.zeros((len(out['t']),))

    if input_mode ==2:
        out['I_cell'][0] = -out['I_max']

    for i in tqdm(range(p['N']-1)):
        f, out['V_cell'][i], out['V_oc'][i], out['SOC_n'][i], out['SOC_p'][i], out['i_s'][i]  = f_SPM(out['x'][:,i],out['I_cell'][i],p)
        out['x'][:,i+1] = out['x'][:,i] + p['dt']*f.reshape((13,))
        if input_mode==1:
            out['I_cell'][i+1] = 0*(out['V_cell'][i] - out['V_ref']) + 0.8*(out['SOC_p'][i]-out['SOC_ref'])
        
        elif input_mode==2:
            if out['V_cell'][i] < out['V_max'] and out['is_CV'][i] ==0:
                out['I_cell'][i+1] = -out['I_max']
            elif out['SOC_p'][i] >= 1:
                out['I_cell'][i] = 0
            else:
                out['is_CV'][i+1]=1
                cs_n, cs_p, Ts, Tc, L_sei, Q, c_solv, c_surf_solv, cs_surf_n, cs_surf_p, T_amb = state_convert(out['x'][:,i+1], p)
                
                f_V_cell_partial = partial(f_V_cell, cs_surf_p,cs_surf_n,Tc, L_sei=L_sei)
                def f_V_I(I):
                    return f_V_cell_partial(I)-out['V_max']
                
                out['I_cell'][i+1] = scipy.optimize.fsolve(f_V_I, out['I_cell'][i])
        
        elif input_mode==3:
            _, _, _, Tc, _, _, _, _, _, _, _ = state_convert(out['x'][:,i+1],p)
            norm_out = normalize_outputs(out['SOC_p'][i], out['V_cell'][i], Tc)
            action = agent.act(norm_out)
            out['I_cell'][i+1] = denormalize_input(action,min_OUTPUT_value,max_OUTPUT_value)

    
    return out

out = SPM(x0, p, input_mode=3)
#Input Mode 0: open loop, 1: feedback, 2: CC-CV, 3: DDPG Actor Output Feedback

out['cs_n'], out['cs_p'], out['Ts'], out['Tc'], out['L_sei'], out['Q'],out['c_solv'], out['c_surf_solv'], out['cs_surf_n'], out['cs_surf_p'], out['T_amb'] = state_convert(out['x'], p)


plt.close('all')
plt.rcParams['text.usetex'] = True

###############################################################################
# Surface Concentrations
###############################################################################

plt.figure(1)
plt.subplot(2,1,1)
plt.plot(out['t'][:],out['cs_surf_n'])
plt.ylabel(r'$cs_n^{surf} [mol.m^{-3}]$')
plt.title('Surface Concentrations')
plt.xlabel('time [s]')
plt.grid()

plt.subplot(2,1,2)
plt.plot(out['t'][:],out['cs_surf_p'])
plt.ylabel(r'$cs_p^{surf} [mol.m^{-3}]$')
plt.xlabel('time [s]')
plt.grid()

plt.tight_layout()
plt.show()

###############################################################################
# Temperatures
###############################################################################

plt.figure(2)
plt.subplot(2,1,1)
plt.plot(out['t'][:],out['Tc'])
plt.ylabel(r'$T_c [^\circ C]$')
plt.title('Core and Surface Temperatures')
plt.xlabel('time [s]')
plt.grid()


plt.subplot(2,1,2)
plt.plot(out['t'][:],out['Ts'])
plt.ylabel(r'$T_s [^\circ C]$')
plt.xlabel('time [s]')
plt.grid()

plt.tight_layout()
plt.show()

###############################################################################
# Side-Reaction Current and Surface Solvent Concentration
###############################################################################

plt.figure(3)
plt.subplot(2,1,1)
plt.plot(out['t'][:-1],out['i_s'])
plt.ylabel(r'$i_s [A.m^{-3}]$')
plt.title('Side-Reaction Current and Solvent Concentration')
plt.xlabel('time [s]')
plt.grid()

plt.subplot(2,1,2)
plt.plot(out['t'][:],out['c_surf_solv'])
plt.ylabel(r'$c_{solv}^{surf} [mol.m^{-3}]$')
plt.xlabel('time [s]')
plt.grid()

plt.tight_layout()
plt.show()

###############################################################################
# SEI Thickness and Capacity
###############################################################################

plt.figure(4)
plt.subplot(2,1,1)
plt.plot(out['t'][:],out['L_sei'])
plt.ylabel(r'$L_{sei} [m]$')
plt.title('SEI Thickness and Battery Capacity')
plt.xlabel('time [s]')
plt.grid()

plt.subplot(2,1,2)
plt.plot(out['t'][:],out['Q'])
plt.ylabel(r'$Q [Ah]$')
plt.xlabel('time [s]')
plt.grid()

plt.tight_layout()
plt.show()

###############################################################################
# Cell Current and Cell Voltage
###############################################################################

plt.figure(5)
plt.subplot(2,1,1)
plt.plot(out['t'][:],out['I_cell'])
plt.ylabel(r'$I_{cell} [A]$')
plt.title('Cell Current and Voltage')
plt.xlabel('time [s]')
plt.grid()

plt.subplot(2,1,2)
plt.plot(out['t'][:-1],out['V_cell'])
plt.ylabel(r'$V [V]$')
plt.xlabel('time [s]')
plt.grid()

plt.tight_layout()
plt.show()

###############################################################################
# SOC
###############################################################################

plt.figure(6)
plt.subplot(2,1,1)
plt.plot(out['t'][:-1],out['SOC_n'])
plt.ylabel(r'$SOC_n$')
plt.title('State of Charge')
plt.xlabel('time [s]')
plt.grid()

plt.subplot(2,1,2)
plt.plot(out['t'][:-1],out['SOC_p'])
plt.ylabel(r'$SOC_p$')
plt.xlabel('time [s]')
plt.grid()

plt.tight_layout()
plt.show()

###############################################################################
# Partition Sequence Generation
###############################################################################

# part = {}

# part['cs_n'] = np.matrix([[0, 5000, 12500, 23230],
#                              [0, 5000, 12500, 23230],
#                              [0, 5000, 12500, 23230]])

# part['cs_p'] = np.matrix([[1.2e4, 1.8e4, 2.2e4, 27362],
#                                [1.2e4, 1.8e4, 2.2e4, 27362],
#                                [1.2e4, 1.8e4, 2.2e4, 27362]])

# part['Ts'] = np.array([-273, 30, 35])
# part['Tc'] = np.array([-273, 30, 35])
# part['L_sei'] = np.array([5e-9, 5e-8, 5e-7, 5e-6])
# part['Q'] = np.array([0, 5, 10, 13])

# part['c_solv'] = np.matrix([[0, 4797, 4799, 4850],
#                                [0, 4797, 4799, 4850],
#                                [0, 4797, 4799, 4850]])

# def assign_partition(x,part,p):
#     n = len(x)
#     s = []
#     for i in range(0,p['nrn']): #cs_n
#         for j in range(part['cs_n'].shape[1]-1):
#             if x[i] >= part['cs_n'][i,j] and x[i] <= part['cs_n'][i,j+1]:
#                 s.append(j)
#                 break
#     for i in range(p['nrn'],p['nrn']+p['nrp']): #cs_p
#         for j in range(part['cs_p'].shape[1]-1):
#             if x[i] >= part['cs_p'][i-p['nrn'],j] and x[i] <= part['cs_p'][i-p['nrn'],j+1]:
#                 s.append(j)
#                 break
    
#     i = p['nrn'] + p['nrp'] #Ts
#     for j in range(part['Ts'].shape[0]):
#         if x[i] >= part['Ts'][j] and x[i] <= part['Ts'][j+1]:
#             s.append(j)
#             break

#     i = p['nrn'] + p['nrp'] + 1 #Tc
#     for j in range(part['Tc'].shape[0]):
#         if x[i] >= part['Tc'][j] and x[i] <= part['Tc'][j+1]:
#             s.append(j)
#             break

#     i = p['nrn'] + p['nrp'] + 2 #L_sei
#     for j in range(part['L_sei'].shape[0]):
#         if x[i] >= part['L_sei'][j] and x[i] <= part['L_sei'][j+1]:
#             s.append(j)
#             break

#     i = p['nrn'] + p['nrp'] + 3 #Q
#     for j in range(part['Q'].shape[0]):
#         if x[i] >= part['Q'][j] and x[i] <= part['Q'][j+1]:
#             s.append(j)
#             break

#     for i in range(p['nrn'] + p['nrp'] + 4, n):
#         for j in range(part['c_solv'].shape[1]-1):
#             if x[i] >= part['c_solv'][i-p['nrn']-p['nrp']-4,j] and x[i] <= part['cs_n'][i-p['nrn']-p['nrp']-4,j+1]:
#                 s.append(j)
#                 break

#     return s

# out = SPM(x0, p, input_mode=2)
# s = []
# for i in range(out['x'].shape[1]):
#     s.append(str(assign_partition(out['x'][:,i], part, p)))

# s_unique = set(s)







