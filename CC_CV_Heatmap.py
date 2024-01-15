import numpy as np
from SPM_Params import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.optimize
import torch
from functools import *

from SPM_Simulator import SPM

I_MAX = np.linspace(0.1*OneC(p), 3.5*OneC(p),10)
V_MAX = np.linspace(3.9,4.3,10)

x0 = np.array([2927,2927,2927,25611,25611,25611,25,25,5e-9,13,4800,4800,4800]).reshape((13,1))

I_S_SUM = np.zeros((len(I_MAX)))
TIME = np.zeros((len(I_MAX)))

for v_max in tqdm(V_MAX):
    I_S_SUM_I = np.array([])
    TIME_I = np.array([])
    for i_max in tqdm(I_MAX):
        p['V_max'] = v_max
        p['I_max'] = i_max

        out = SPM(x0,p,input_mode=2)
        I_S_SUM_I = np.hstack((I_S_SUM_I, p['dt']*np.sum(np.abs(out['i_s']))))
        TIME_I = np.hstack((TIME_I, out['t'][np.argmax(out['SOC_p']>0.80)]))

    I_S_SUM = np.vstack((I_S_SUM_I, I_S_SUM))
    TIME = np.vstack((TIME_I, TIME))

I_S_SUM = I_S_SUM[:-1]
TIME = TIME[:-1]

nx = I_MAX.shape[0]
no_labels_x = nx # how many labels to see on axis x
step_x = int(nx / (no_labels_x - 1)) # step between consecutive labels
x_positions = np.arange(0,nx,step_x) # pixel count at label position

ny = V_MAX.shape[0]
no_labels_y = ny # how many labels to see on axis x
step_y = int(ny / (no_labels_y - 1)) # step between consecutive labels
y_positions = np.arange(0,ny,step_x) # pixel count at label position


plt.imshow(I_S_SUM, interpolation='nearest')
plt.colorbar()
plt.xticks(x_positions, np.trunc(100*I_MAX/OneC(p))/100)
plt.yticks(y_positions, np.trunc(100*np.flip(V_MAX))/100)
plt.tight_layout()
plt.title('Total Side-Reaction Current Density')
plt.xlabel('i_max (C)')
plt.ylabel('v_max (V)')
plt.show()

plt.imshow(TIME, interpolation='nearest')
plt.colorbar()
plt.xticks(x_positions, np.trunc(100*I_MAX/OneC(p))/100)
plt.yticks(y_positions, np.trunc(100*np.flip(V_MAX))/100)
plt.tight_layout()
plt.title('Time to 80% SOC')
plt.xlabel('i_max (C)')
plt.ylabel('v_max (V)')
plt.show()










