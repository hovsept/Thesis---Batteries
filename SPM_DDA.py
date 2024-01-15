#SPM Data-Driven Abstraction
#Hovsep Touloujian - Dec 8th 2023
import numpy as np
from SPM_Params import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.optimize
import torch
from functools import *
import time
import heapq

from ddpg_agent import Agent
from SPM_Simulator import SPM
from scenario_epsilon import eps_general

def trajectory(x0,p, H):
    p['N'] = H+2
    p['Tf'] = (p['N']-1)*p['dt']
    #0: open loop, 1: feedback, 2: CC-CV, 3: DDPG Actor Output Feedback, 4: Discharge
    out = SPM(x0,p,input_mode=3, start_episode=30)
    traj = np.vstack((out['SOC_p'],out['i_s']))
    while traj.shape[-1]<H:
        traj_x = np.zeros((traj.shape[0], traj.shape[1]+1))
        for i in range(traj.shape[0]):
           traj_x[i] = np.append(traj[i],traj[i,-1])
        traj = traj_x
        
    return traj

H = 20000
N = 250
n_vars = 2
all_trajs = np.zeros((N,n_vars,H))

print('-----------------------------------------------------')
print('Generating N =', N, 'trajectories of length H =', H)
print('-----------------------------------------------------')

max_i_s = -5e-6
SOC_threshold = 0.9

for i in tqdm(range(N)):
    # x0 = set_sample(np.abs(max_i_s),0.,1.,0)
    x0 = sample_IC(p)
    all_trajs[i,:,:] = trajectory(x0,p,H)


#0: Safe Set
#1: Unsafe Set
#2: Goal Set

def direct_partition(all_trajs, max_i_s, SOC_threshold):
    all_trajs_part = np.zeros((N,H))
    for i in tqdm(range(N)):
        for j in range(H):
            if all_trajs[i,0,j] >= SOC_threshold and all_trajs[i,1,j]>= max_i_s:
                all_trajs_part[i,j] = 2.
            elif all_trajs[i,0,j] <= SOC_threshold and all_trajs[i,1,j]>= max_i_s:
                all_trajs_part[i,j] = 0.
            else:
                all_trajs_part[i,j] = 1.

    return all_trajs_part

print('-----------------------------------------------------')
print('Partitioning Trajectories')
print('-----------------------------------------------------')
all_trajs_part = direct_partition(all_trajs,max_i_s,SOC_threshold)

viol_traj = []
for i in range(N):
    if all_trajs_part[i][0] == 0 and 1 in all_trajs_part[i]:
        viol_traj.append(i)


ell = 500

def get_ell_sequences(all_trajs_part, ell,H):
    ell_seq_trajectory = set()
    ell_seq_init = set()
    for trajectory_parts in tqdm(all_trajs_part):
        idx = 0
        for idx in range(0, H-ell+1):
            ell_seq_trajectory.add( tuple(trajectory_parts[idx:idx+ell]) )
        # find all ell-seq from INITIAL STATE
        ell_seq_init.add(tuple(trajectory_parts[0:ell]))
        # find ONE ell-seq from a trajectory at a random point

    return ell_seq_trajectory, ell_seq_init

print('-----------------------------------------------------')
print('Generating ell-Sequences ell = ', str(ell))
print('-----------------------------------------------------')
ell_seq_trajectory, ell_seq_init = get_ell_sequences(all_trajs_part, ell, H)

if len(ell_seq_trajectory) > len(ell_seq_init):
    print(f'Visited ell-sequences are more than the initial ones: \n'
          f'visited {len(ell_seq_trajectory)}, initial: {len(ell_seq_init)}.')
elif len(ell_seq_trajectory) > len(ell_seq_init):
    print(f'Randomly picked ell-sequences == visited partitions: \n'
          f'visited {len(ell_seq_trajectory)}, initial: {len(ell_seq_init)}.')
else:
    print(f'Same number of seen and initial sequences: ({len(ell_seq_init)}).')

def greedy_set_cover(subsets: set, parent_set: set):
    #parent_set = set(e for s in parent_set for e in s)
    max = len(parent_set)
    # create the initial heap.
    # Note 'subsets' can be unsorted,
    # so this is independent of whether remove_redunant_subsets is used.
    heap = []
    for s in subsets:
        # Python's heapq lets you pop the *smallest* value, so we
        # want to use max-len(s) as a score, not len(s).
        # len(heap) is just proving a unique number to each subset,
        # used to tiebreak equal scores.
        heapq.heappush(heap, [max-len(s), len(heap), s])
    #results = []
    result_set = set()
    num_sets = 0
    u = 1
    tic = time.perf_counter()
    while result_set < parent_set:
        #logging.debug('len of result_set is {0}'.format(len(result_set)))
        best = []
        unused = []
        while heap:
            score, count, s = heapq.heappop(heap)
            if not best:
                best = [max-len(s - result_set), count, s]
                continue
            if score >= best[0]:
                # because subset scores only get worse as the resultset
                # gets bigger, we know that the rest of the heap cannot beat
                # the best score. So push the subset back on the heap, and
                # stop this iteration.
                heapq.heappush(heap, [score, count, s])
                break
            score = max-len(s - result_set)
            if score >= best[0]:
                unused.append([score, count, s])
            else:
                unused.append(best)
                best = [score, count, s]
        add_set = best[2]
        #logging.debug('len of add_set is {0} score was {1}'.format(len(add_set), best[0]))
        #results.append(add_set)
        result_set.update(add_set)
        num_sets += 1
        # subsets that were not the best get put back on the heap for next time.
        while unused:
            heapq.heappush(heap, unused.pop())
        if (len(result_set) / (u*2000)) > 1:
            toc = time.perf_counter()
            # Print percentage of covered elements
            print(f'{len(result_set)/len(parent_set)*100:.2f}%')
            u += 1
            print(f'Elapsed time: {toc - tic:0.4f} seconds')
            tic = toc
    return num_sets

num_sets = 0
if ell < H:
    # Recast for set cover problem
    subsets = []
    for H_seq in all_trajs_part:
        seq_of_ell_seq = []
        for i in range(0, len(H_seq)-ell+1):
            seq_of_ell_seq.append(tuple(H_seq[i:i+ell]))
        subsets.append(set(seq_of_ell_seq))
    tic = time.perf_counter()
    num_sets = greedy_set_cover(subsets, ell_seq_trajectory)
    toc = time.perf_counter()
    print(f"Time elapsed: {toc - tic:0.4f} seconds")
else:
    num_sets = len(ell_seq_trajectory)



print("Upper bound of complexity ", num_sets)

print('-'*80)
epsi_up = eps_general(k=num_sets, N=N, beta=1e-12)
print(f'Epsilon Bound using complexity: {epsi_up}')
print('-'*80)

################################################
# 2D-Visualization for SOC and i_s trajectories
################################################

fig, ax = plt.subplots()
for seq in all_trajs[:]:
    ax.plot(seq[0],seq[1])
    ax.fill_between(seq[0],4.75*max_i_s, max_i_s, where=seq[0]>=0, color = "lightcoral")
    ax.fill_between(seq[0],max_i_s,1e-6, where = seq[0]>=SOC_threshold, color = "palegreen")

ax.set_xlabel('SOC')
ax.set_ylabel('i_s')
ax.set_title("N = " + str(N))
fig.tight_layout()

################################################
# Counterexamples
################################################

fig, ax = plt.subplots()
for seq in all_trajs[viol_traj]:
    if seq[1][-1]< seq[1][0]:
        ax.plot(seq[0],seq[1])
        ax.fill_between(seq[0],4.75*max_i_s, max_i_s, where=seq[0]>=0, color = "lightcoral")
        ax.fill_between(seq[0],max_i_s,1e-6, where = seq[0]>=SOC_threshold, color = "palegreen")

ax.set_xlabel('SOC')
ax.set_ylabel('i_s')
ax.set_title("Number of Counterexamples: "+ str(len(viol_traj)))
fig.tight_layout()


