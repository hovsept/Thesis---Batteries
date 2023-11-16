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
from functools import *

x0 = sample_IC(p)
x0 = np.array([2927,2927,2927,25611,25611,25611,25,25,5e-9,13,4800,4800,4800]).reshape((13,1))
input_mode = 2 #0: open loop, 1: feedback, 2: CC-CV

out = {}
out['t'] = np.arange(0,p['Tf'],p['dt'])

out['I_cell'] = np.zeros((len(out['t'])))
if input_mode == 0:
    out['I_cell'] = -0.3*np.ones((len(out['t'],)))
elif input_mode==1:
    out['V_ref'] = 4.1
    out['SOC_ref'] = 0.9
    out['I_cell'][0] = 0
elif input_mode==2:
    out['V_max'] = 4.1
    out['I_max'] = 1
    out['I_min'] = 0.1
    
    
n = p['nrn']+p['nrp']+p['n_sei']+4
out['x'] = np.zeros((n,len(out['t'])))
out['x'][:,0] = x0.reshape((n,))
out['V_cell'] = np.zeros((len(out['t'])-1,))
out['V_oc'] = np.zeros((len(out['t'])-1,))
out['SOC_n'] = np.zeros((len(out['t'])-1,))
out['SOC_p'] = np.zeros((len(out['t'])-1,))
out['i_s'] = np.zeros((len(out['t'])-1,))
out['is_CV'] = np.zeros((len(out['t']),))


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

            
        

out['cs_n'], out['cs_p'], out['Ts'], out['Tc'], out['L_sei'], out['Q'],out['c_solv'], out['c_surf_solv'], out['cs_surf_n'], out['cs_surf_p'], out['T_amb'] = state_convert(out['x'], p)

plt.close('all')
plt.rcParams['text.usetex'] = True

###############################################################################
# Surface Concentrations
###############################################################################

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






