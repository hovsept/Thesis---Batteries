# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 16:22:09 2023

@author: Hovsep Touloujian
"""

import numpy as np
from SPM_Params import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

import scipy.optimize
import torch
from functools import *
from settings import *


# import sys
# sys.path.append('RL-Batt-DDPG-main\OutputFeedback_RL')

from ddpg_agent import Agent

x0 = sample_IC(p)
# x0 = np.array([2927,2927,2927,25611,25611,25611,25,25,5e-9,13,4800,4800,4800]).reshape((13,1))
# x0 = set_sample(2.5e-5,0.9,1.0,0)
# cs_n0, cs_p0, Ts0, Tc0, L_sei0, Q0, c_solv0, c_surf_solv0, cs_surf_n0, cs_surf_p0, T_amb = state_convert(x0,p)

def SPM(x0,p,input_mode = 3, start_episode = 30):
     #0: open loop, 1: feedback, 2: CC-CV, 3: DDPG Actor Output Feedback, 4: Discharge

    out = {}

    out['I_cell'] = np.zeros((p['N']))
    if input_mode == 0:
        out['I_cell'] = -OneC(p)*np.ones((len(out['t'],)))
    elif input_mode==1:
        out['V_ref'] = 4.1
        out['SOC_ref'] = 0.9
        out['I_cell'][0] = 0
    elif input_mode == 3:
        i_training = 0
        agent = Agent(state_size=3, action_size=1, random_seed=i_training)  # the number of state is 496.

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

        min_OUTPUT_value = control_settings['max_charging_current']
        max_OUTPUT_value = 0
    # compute actual action from normalized action 
        def denormalize_input(input_value, min_OUTPUT_value, max_OUTPUT_value):
            output_value=(1+input_value)*(max_OUTPUT_value-min_OUTPUT_value)/2+min_OUTPUT_value
            return output_value
        
    elif input_mode == 4:
            p['I_discharge'] = OneC(p)

        
        
    n = p['nrn']+p['nrp']+p['n_sei']+4
    out['x'] = np.zeros((n,p['N']))
    out['x'][:,0] = x0.reshape((n,))
    out['V_cell'] = np.zeros((p['N']-1,))
    out['V_oc'] = np.zeros((p['N']-1,))
    out['SOC_n'] = np.zeros((p['N']-1,))
    out['SOC_p'] = np.zeros((p['N']-1,))
    out['i_s'] = np.zeros((p['N']-1,))
    out['is_CV'] = np.zeros((p['N'],))

    if input_mode ==2:
        out['I_cell'][0] = -p['I_max']
    elif input_mode == 4:
        out['I_cell'][0] = p['I_discharge']

    # for i in tqdm(range(p['N']-1)):
    for i in range(p['N']-1):
        f, out['V_cell'][i], out['V_oc'][i], out['SOC_n'][i], out['SOC_p'][i], out['i_s'][i]  = f_SPM(out['x'][:,i],out['I_cell'][i],p)
        out['x'][:,i+1] = out['x'][:,i] + p['dt']*f.reshape((13,))
        if input_mode==1:
            out['I_cell'][i+1] = 0*(out['V_cell'][i] - out['V_ref']) + 0.8*(out['SOC_p'][i]-out['SOC_ref'])
        
        elif input_mode==2:
            if out['V_cell'][i] < p['V_max'] and out['is_CV'][i] ==0:
                out['I_cell'][i+1] = -p['I_max']
            elif out['SOC_p'][i] >= 1:
                out['I_cell'][i] = 0
            else:
                out['is_CV'][i+1]=1
                cs_n, cs_p, Ts, Tc, L_sei, Q, c_solv, c_surf_solv, cs_surf_n, cs_surf_p, T_amb = state_convert(out['x'][:,i+1], p)
                
                f_V_cell_partial = partial(f_V_cell, cs_surf_p,cs_surf_n,Tc, L_sei=L_sei)
                def f_V_I(I):
                    return f_V_cell_partial(I)-p['V_max']
                
                out['I_cell'][i+1] = scipy.optimize.fsolve(f_V_I, out['I_cell'][i])
        
        elif input_mode==3:
            _, _, _, Tc, _, _, _, _, _, _, _ = state_convert(out['x'][:,i+1],p)
            norm_out = normalize_outputs(out['SOC_p'][i], out['V_cell'][i], Tc)
            action = agent.act(norm_out, add_noise = False)
            out['I_cell'][i+1] = denormalize_input(action,min_OUTPUT_value,max_OUTPUT_value)

        elif input_mode==4:
            out['I_cell'][i+1] = p['I_discharge']
            if out['SOC_n'][i]<=0.05 or out['SOC_p'][i]<=0.05:
                out['t'] = np.arange(0,i*p['dt'],p['dt'])
                break

        if out['SOC_n'][i]>=0.99 or out['SOC_p'][i]>=0.99:
            out['t'] = np.arange(0,i*p['dt'],p['dt'])
            break
    
    if out['SOC_n'][i]<0.99 and out['SOC_p'][i]<0.99 and input_mode != 4:
        out['t'] = np.arange(0,p['Tf'],p['dt'])

    out['I_cell'] = out['I_cell'][:len(out['t'])]
    out['x'] = out['x'][:,:len(out['t'])]
    out['x'][:,0] = x0.reshape((n,))
    out['V_cell'] = out['V_cell'][:len(out['t'])-1]
    out['V_oc'] = out['V_oc'][:len(out['t'])-1]
    out['SOC_n'] = out['SOC_n'][:len(out['t'])-1]
    out['SOC_p'] = out['SOC_p'][:len(out['t'])-1]
    out['i_s'] = out['i_s'][:len(out['t'])-1]
    out['is_CV'] = out['is_CV'][:len(out['t'])]

    out['cs_n'], out['cs_p'], out['Ts'], out['Tc'], out['L_sei'], out['Q'],out['c_solv'], out['c_surf_solv'], out['cs_surf_n'], out['cs_surf_p'], out['T_amb'] = state_convert(out['x'], p)
    return out


out = SPM(x0, p, input_mode=3, start_episode = 30)
#Input Mode 0: open loop, 1: feedback, 2: CC-CV, 3: DDPG Actor Output Feedback



# plt.close('all')
# plt.rcParams['text.usetex'] = True

# ###############################################################################
# # Surface Concentrations
# ###############################################################################

# plt.figure(1)
# plt.subplot(2,1,1)
# plt.plot(out['t'][:],out['cs_surf_n'])
# plt.ylabel(r'$cs_n^{surf} [mol.m^{-3}]$')
# plt.title('Surface Concentrations')
# plt.xlabel('time [s]')
# plt.grid()

# plt.subplot(2,1,2)
# plt.plot(out['t'][:],out['cs_surf_p'])
# plt.ylabel(r'$cs_p^{surf} [mol.m^{-3}]$')
# plt.xlabel('time [s]')
# plt.grid()

# plt.tight_layout()
# plt.show()

# ###############################################################################
# # Temperatures
# ###############################################################################

# plt.figure(2)
# plt.subplot(2,1,1)
# plt.plot(out['t'][:],out['Tc'])
# plt.ylabel(r'$T_c [^\circ C]$')
# plt.title('Core and Surface Temperatures')
# plt.xlabel('time [s]')
# plt.grid()


# plt.subplot(2,1,2)
# plt.plot(out['t'][:],out['Ts'])
# plt.ylabel(r'$T_s [^\circ C]$')
# plt.xlabel('time [s]')
# plt.grid()

# plt.tight_layout()
# plt.show()

# ###############################################################################
# # Side-Reaction Current and Surface Solvent Concentration
# ###############################################################################

# plt.figure(3)
# plt.subplot(2,1,1)
# plt.plot(out['t'][:-1],out['i_s'])
# plt.ylabel(r'$i_s [A.m^{-3}]$')
# plt.title('Side-Reaction Current and Solvent Concentration')
# plt.xlabel('time [s]')
# plt.grid()

# plt.subplot(2,1,2)
# plt.plot(out['t'][:],out['c_surf_solv'])
# plt.ylabel(r'$c_{solv}^{surf} [mol.m^{-3}]$')
# plt.xlabel('time [s]')
# plt.grid()

# plt.tight_layout()
# plt.show()

# ###############################################################################
# # SEI Thickness and Capacity
# ###############################################################################

# plt.figure(4)
# plt.subplot(2,1,1)
# plt.plot(out['t'][:],out['L_sei'])
# plt.ylabel(r'$L_{sei} [m]$')
# plt.title('SEI Thickness and Battery Capacity')
# plt.xlabel('time [s]')
# plt.grid()

# plt.subplot(2,1,2)
# plt.plot(out['t'][:],out['Q'])
# plt.ylabel(r'$Q [Ah]$')
# plt.xlabel('time [s]')
# plt.grid()

# plt.tight_layout()
# plt.show()

# ###############################################################################
# # Cell Current and Cell Voltage
# ###############################################################################

# plt.figure(5)
# plt.subplot(2,1,1)
# plt.plot(out['t'][:],out['I_cell'])
# plt.ylabel(r'$I_{cell} [A]$')
# plt.title('Cell Current and Voltage')
# plt.xlabel('time [s]')
# plt.grid()

# plt.subplot(2,1,2)
# plt.plot(out['t'][:-1],out['V_cell'])
# plt.ylabel(r'$V [V]$')
# plt.xlabel('time [s]')
# plt.grid()

# plt.tight_layout()
# plt.show()

# ###############################################################################
# # SOC
# ###############################################################################

# plt.figure(6)
# plt.subplot(2,1,1)
# plt.plot(out['t'][:-1],out['SOC_n'])
# plt.ylabel(r'$SOC_n$')
# plt.title('State of Charge')
# plt.xlabel('time [s]')
# plt.grid()

# plt.subplot(2,1,2)
# plt.plot(out['t'][:-1],out['SOC_p'])
# plt.ylabel(r'$SOC_p$')
# plt.xlabel('time [s]')
# plt.grid()

# plt.tight_layout()
# plt.show()








