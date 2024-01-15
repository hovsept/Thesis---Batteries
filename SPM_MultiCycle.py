import numpy as np
from SPM_Params import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from functools import *

from SPM_Simulator import SPM

from ddpg_agent import Agent

N_cycles = 10

def SPM_cycle(N_cycles):
    out_cycle = {}
    out_cycle['t_cycle'] = [0.]

    x0 = np.array([2927,2927,2927,25611,25611,25611,25,25,5e-9,13,4800,4800,4800]).reshape((13,1))
    out_cycle['x'] = x0
    out_cycle['t'] = 0.
    for i in tqdm(range(N_cycles)):
        #Charge
        out = SPM(x0,p,input_mode=2)
        out_cycle['x'] = np.hstack((out_cycle['x'], out['x'][:,1:]))
        out_cycle['t'] = np.hstack((out_cycle['t'],out['t'][1:] + out_cycle['t_cycle'][-1]))

        out_cycle['t_cycle'].append(out_cycle['t'][-1])
        x0 = out['x'][:,-1].reshape((13,1))

        if i==0:
            out_cycle['I_cell'] = out['I_cell']
            out_cycle['V_cell'] = out['V_cell']
            out_cycle['V_oc'] = out['V_oc']
            out_cycle['i_s'] = out['i_s']
            out_cycle['SOC_n'] = out['SOC_n']
            out_cycle['SOC_p'] = out['SOC_p']
        else:
            out_cycle['I_cell'] = np.hstack((out_cycle['I_cell'], out['I_cell'][1:]))
            out_cycle['V_cell'] = np.hstack((out_cycle['V_cell'], out['V_cell'][:]))
            out_cycle['V_oc'] = np.hstack((out_cycle['V_oc'], out['V_oc'][:]))
            out_cycle['i_s'] = np.hstack((out_cycle['i_s'], out['i_s'][:]))
            out_cycle['SOC_n'] = np.hstack((out_cycle['SOC_n'], out['SOC_n'][:]))
            out_cycle['SOC_p'] = np.hstack((out_cycle['SOC_p'], out['SOC_p'][:]))


        #Discharge
        out = SPM(x0,p,input_mode=4)
        out_cycle['x'] = np.hstack((out_cycle['x'], out['x'][:,1:]))
        out_cycle['t'] = np.hstack((out_cycle['t'],out['t'][1:] + out_cycle['t_cycle'][-1]))

        out_cycle['t_cycle'].append(out_cycle['t'][-1])
        x0 = out['x'][:,-1].reshape((13,1))

        out_cycle['I_cell'] = np.hstack((out_cycle['I_cell'], out['I_cell'][1:]))
        out_cycle['V_cell'] = np.hstack((out_cycle['V_cell'], out['V_cell'][:]))
        out_cycle['V_oc'] = np.hstack((out_cycle['V_oc'], out['V_oc'][:]))
        out_cycle['i_s'] = np.hstack((out_cycle['i_s'], out['i_s'][:]))
        out_cycle['SOC_n'] = np.hstack((out_cycle['SOC_n'], out['SOC_n'][:]))
        out_cycle['SOC_p'] = np.hstack((out_cycle['SOC_p'], out['SOC_p'][:]))

    out_cycle['cs_n'], out_cycle['cs_p'], out_cycle['Ts'],out_cycle['Tc'], out_cycle['L_sei'], out_cycle['Q'],out_cycle['c_solv'],out_cycle['c_surf_solv'], out_cycle['cs_surf_n'], out_cycle['cs_surf_p'],out_cycle['T_amb'] = state_convert(out_cycle['x'], p)

    return out_cycle, out

out_cycle = SPM_cycle(N_cycles)

out_cycle = out_cycle[0]
plt.close('all')
plt.rcParams['text.usetex'] = True

###############################################################################
# Surface Concentrations
###############################################################################

plt.figure(1)
plt.subplot(2,1,1)
plt.plot(out_cycle['t'][:],out_cycle['cs_surf_n'])
plt.ylabel(r'$cs_n^{surf} [mol.m^{-3}]$')
plt.title('Surface Concentrations')
plt.xlabel('time [s]')
plt.grid()

plt.subplot(2,1,2)
plt.plot(out_cycle['t'][:],out_cycle['cs_surf_p'])
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
plt.plot(out_cycle['t'][:],out_cycle['Tc'])
plt.ylabel(r'$T_c [^\circ C]$')
plt.title('Core and Surface Temperatures')
plt.xlabel('time [s]')
plt.grid()


plt.subplot(2,1,2)
plt.plot(out_cycle['t'][:],out_cycle['Ts'])
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
plt.plot(out_cycle['t'][:-1],out_cycle['i_s'])
plt.ylabel(r'$i_s [A.m^{-3}]$')
plt.title('Side-Reaction Current and Solvent Concentration')
plt.xlabel('time [s]')
plt.grid()

plt.subplot(2,1,2)
plt.plot(out_cycle['t'][:],out_cycle['c_surf_solv'])
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
plt.plot(out_cycle['t'][:],out_cycle['L_sei'])
plt.ylabel(r'$L_{sei} [m]$')
plt.title('SEI Thickness and Battery Capacity')
plt.xlabel('time [s]')
plt.grid()

plt.subplot(2,1,2)
plt.plot(out_cycle['t'][:],out_cycle['Q'])
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
plt.plot(out_cycle['t'][:],out_cycle['I_cell'])
plt.ylabel(r'$I_{cell} [A]$')
plt.title('Cell Current and Voltage')
plt.xlabel('time [s]')
plt.grid()

plt.subplot(2,1,2)
plt.plot(out_cycle['t'][:-1],out_cycle['V_cell'])
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
plt.plot(out_cycle['t'][:-1],out_cycle['SOC_n'])
plt.ylabel(r'$SOC_n$')
plt.title('State of Charge')
plt.xlabel('time [s]')
plt.grid()

plt.subplot(2,1,2)
plt.plot(out_cycle['t'][:-1],out_cycle['SOC_p'])
plt.ylabel(r'$SOC_p$')
plt.xlabel('time [s]')
plt.grid()

plt.tight_layout()
plt.show()

