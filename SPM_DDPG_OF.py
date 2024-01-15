import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from tqdm import tqdm

from ddpg_agent import Agent
import pdb
import ipdb

import logz
import scipy.signal


from SPM_env import *
import statistics
import pickle
import os

import argparse

import time

def normalize_outputs(soc, voltage, temperature):
    norm_soc = soc-0.5
    norm_voltage = (voltage-3.5)/1.2
    norm_temperature = (temperature-25-10)/(320-298)
    norm_output = np.array([norm_soc, norm_voltage, norm_temperature],dtype = np.float32)

    return norm_output.transpose()

def denormalize_input(input_value, min_OUTPUT_value, max_OUTPUT_value):
    output_value=(1+input_value)*(max_OUTPUT_value-min_OUTPUT_value)/2+min_OUTPUT_value
    
    return output_value

def eval_policy(policy, eval_episodes = 10):
    eval_env = SPM(sett=settings, cont_sett=control_settings)

    avg_reward = 0.
    avg_temp_vio = 0.
    avg_i_s_vio = 0.
    avg_volt_vio = 0.
    avg_chg_time = 0.

    for _ in range(eval_episodes):
        x_t, done = eval_env.reset(), False
        _, cs_p, _, Tc, _, _, _, _, cs_surf_n, cs_surf_p, _ = state_convert(x_t,p)
        soc = f_SOC_p(cs_p)
        V = f_V_cell(cs_surf_p,cs_surf_n,Tc,0,0) #Assuming I(0)= 0
        norm_out = normalize_outputs(soc,V,Tc)

        ACTION_VEC = []
        T_VEC = []
        V_VEC = []
        I_S_VEC = []
        score = 0
        while not done:
            norm_action = policy.act(norm_out, add_noise = False)
            applied_action = denormalize_input(norm_action, 
                                               eval_env.action_space().low[0],
                                               eval_env.action_space().high[0])
            
            try:
                x_tp1, s_tp1, reward, done = eval_env.step(x_t,applied_action)
            except:
                ipdb.set_trace()

            next_V, next_soc, next_Tc, next_i_s = s_tp1[0], s_tp1[1], s_tp1[2], s_tp1 [3]
            norm_next_out = normalize_outputs(next_soc, next_V, next_Tc)

            ACTION_VEC.append(applied_action)
            T_VEC.append(next_Tc)
            V_VEC.append(next_V)
            I_S_VEC.append(next_i_s)
            score += reward
            x_t = x_tp1
            norm_out = norm_next_out

        avg_reward += score
        avg_temp_vio += np.max(np.array(T_VEC))
        avg_i_s_vio += np.min(np.array(I_S_VEC))
        avg_volt_vio += np.max(np.array(V_VEC))
        avg_chg_time += len(ACTION_VEC)

    avg_reward /= eval_episodes
    avg_temp_vio /= eval_episodes
    avg_volt_vio /= eval_episodes
    avg_chg_time /= eval_episodes
    avg_i_s_vio /= eval_episodes

    avg_MAX_volt_vio = avg_volt_vio
    avg_MAX_temp_vio = avg_temp_vio
    avg_MAX_i_s_vio = avg_i_s_vio

    print("---------------------------------------")
    print("Evaluation over {} episodes: {:.3f}".format(eval_episodes, avg_reward[0]))
    print("Average i_s: ", avg_i_s_vio)
    print("---------------------------------------")

    policy_heatmap(policy, T = 25)

    return avg_reward, avg_MAX_temp_vio, avg_MAX_volt_vio, avg_chg_time


# function for training ddpg agent

def ddpg(n_episodes = 3000, i_training = 1, start_episode = 0):
    scores_list = []
    checkpoints_list = []

    #Save the initial parameters of actor-critic networks
    i_episode = start_episode
    checkpoints_list.append(i_episode)
    try:
        os.makedirs('results/SPM_training_results/training'+str(i_training)+'/episode'+str(i_episode))
    except:
        pass

    torch.save(agent.actor_local.state_dict(), 'results/SPM_training_results/training'+str(i_training)+'/episode'+str(i_episode)+'/checkpoint_actor_'+str(i_episode)+'.pth')
    torch.save(agent.critic_local.state_dict(), 'results/SPM_training_results/training'+str(i_training)+'/episode'+str(i_episode)+'/checkpoint_critic_'+str(i_episode)+'.pth')
    torch.save(agent.actor_optimizer.state_dict(), 'results/SPM_training_results/training'+str(i_training)+'/episode'+str(i_episode)+'/checkpoint_actor_optimizer_'+str(i_episode)+'.pth')
    torch.save(agent.critic_optimizer.state_dict(), 'results/SPM_training_results/training'+str(i_training)+'/episode'+str(i_episode)+'/checkpoint_critic_optimizer_'+str(i_episode)+'.pth')


    # Evaluate the initial (untrained) policy
    print('Evaluate first')
    # evaluations = [eval_policy(agent)]
    evaluations = []
    # ipdb.set_trace()
    # evaluations = []

    for i_episode in range(start_episode, n_episodes+1):

        #random initial values
        x_t = env.reset()
        _, cs_p, _, Tc, _, _, _, _, cs_surf_n, cs_surf_p, _ = state_convert(x_t,p)
        soc = f_SOC_p(cs_p)
        soc0 = soc
        V = f_V_cell(cs_surf_p,cs_surf_n,Tc,0,0) #Assuming I(0)= 0
        norm_out = normalize_outputs(soc[0],V,Tc)

        agent.reset()

        score = 0
        done = False
        cont = 0

        V_VEC = []
        T_VEC = []
        SOC_VEC = []
        CURRENT_VEC = []
        I_S_VEC = []

        while not done or score > control_settings['max_negative_score']:

            norm_action = agent.act(norm_out, add_noise = True)
            applied_action = denormalize_input(norm_action,
                                               env.action_space().low[0],
                                               env.action_space().high[0])
            
            #apply action
            x_tp1, s_tp1, reward, done = env.step(x_t,applied_action)
            
            next_V, next_soc, next_Tc, next_i_s = s_tp1[0], s_tp1[1], s_tp1[2], s_tp1 [3]
            norm_next_out = normalize_outputs(next_soc, next_V, next_Tc)

            V_VEC.append(next_V)
            T_VEC.append(next_Tc)
            SOC_VEC.append(next_soc)
            CURRENT_VEC.append(applied_action)
            I_S_VEC.append(next_i_s)
            # print(i_episode, next_soc)

            #update agent, add to replay buffer
            agent.step(norm_out, norm_action, reward, norm_next_out, done)

            try:
                score += reward
            except:
                pdb.set_trace()
            cont+=1
            if done:
                break

            norm_out = norm_next_out
            x_t = x_tp1

        print("Training", i_training)
        print("\rEpisode number", i_episode)
        print("reward: ", score)
        print("Average Cell Current: ", sum(CURRENT_VEC)/len(CURRENT_VEC))
        print("max i_s: ", min(I_S_VEC))
        print("Initial SOC: ", soc0)
        print("Simulation Time: ", env.episode_step*p['dt'])


        scores_list.append(score)

        if (i_episode % settings['periodic_test']) == 0:
        #save the checkpoint for actor, critic and optimizer (for loading the agent)
            checkpoints_list.append(i_episode)
            try:
                os.makedirs('results/SPM_training_results/training'+str(i_training)+'/episode'+str(i_episode))
            except:
                pass
            
            torch.save(agent.actor_local.state_dict(), 'results/SPM_training_results/training'+str(i_training)+'/episode'+str(i_episode)+'/checkpoint_actor_'+str(i_episode)+'.pth')
            torch.save(agent.critic_local.state_dict(), 'results/SPM_training_results/training'+str(i_training)+'/episode'+str(i_episode)+'/checkpoint_critic_'+str(i_episode)+'.pth')
            torch.save(agent.actor_optimizer.state_dict(), 'results/SPM_training_results/training'+str(i_training)+'/episode'+str(i_episode)+'/checkpoint_actor_optimizer_'+str(i_episode)+'.pth')
            torch.save(agent.critic_optimizer.state_dict(), 'results/SPM_training_results/training'+str(i_training)+'/episode'+str(i_episode)+'/checkpoint_critic_optimizer_'+str(i_episode)+'.pth')

        if (i_episode % settings['periodic_test']) == 0 and i_episode>0:
            # Perform evaluation test
            evaluations.append(eval_policy(agent))
            try:
                os.makedirs('results/SPM_testing_results/training'+str(i_training))
            except:
                pass
            
            np.save('results/SPM_testing_results/training'+str(i_training)+'/eval.npy',evaluations)

    return scores_list, checkpoints_list

#-------------------------------------------------------------------------------------
#Policy Heatmap Illustration

def policy_heatmap(agent, T = 25):
    print("------------------------------------------------")
    print("Generating Heatmap of Policy with T = "+ str(T))
    print("------------------------------------------------")

    SOC_grid = np.linspace(0,1,10)
    V_grid = np.linspace(3.7,4.3,10)

    ACTION = np.zeros((len(SOC_grid)))

    for V in tqdm(V_grid):
        ACTION_I = np.array([])
        for soc in SOC_grid:

            norm_out = normalize_outputs(soc,V,T)
            action = agent.act(norm_out, add_noise = False)
            applied_action = denormalize_input(action, env.action_space().low[0], env.action_space().high[0])

            ACTION_I = np.hstack((ACTION_I, applied_action))

        ACTION = np.vstack((ACTION_I, ACTION))

    ACTION = ACTION[:-1]

    nx = SOC_grid.shape[0]
    no_labels_x = nx # how many labels to see on axis x
    step_x = int(nx / (no_labels_x - 1)) # step between consecutive labels
    x_positions = np.arange(0,nx,step_x) # pixel count at label position

    ny = V_grid.shape[0]
    no_labels_y = ny # how many labels to see on axis x
    step_y = int(ny / (no_labels_y - 1)) # step between consecutive labels
    y_positions = np.arange(0,ny,step_x) # pixel count at label position


    plt.imshow(ACTION, interpolation='nearest')
    plt.colorbar()
    plt.xticks(x_positions, np.trunc(100*SOC_grid)/100)
    plt.yticks(y_positions, np.trunc(100*np.flip(V_grid))/100)
    plt.tight_layout()
    plt.title('RL Policy, T = ' + str(T))
    plt.xlabel('SOC')
    plt.ylabel('Voltage (V)')
    plt.show(block=True)
    return ACTION
#-------------------------------------------------------------------------------------
#MAIN

initial_conditions={}

# assign the environment
env = SPM(sett=settings, cont_sett=control_settings)

# parser
parser = argparse.ArgumentParser()
parser.add_argument('--id', type = int)
# args = parser.parse_args()
args, unknown = parser.parse_known_args()

# Seeding
i_seed = args.id
i_seed = 0
i_training = i_seed
np.random.seed(i_seed)
torch.manual_seed(i_seed)


#TRAINING simulation
total_returns_list_with_exploration=[]

#-------------------------------------------------------------------------------------
#assign the agent which is a ddpg
agent = Agent(state_size=3, action_size=1, random_seed=i_training)  # the number of state is 496.

start_episode = 200
if start_episode !=0:
    agent.actor_local.load_state_dict(torch.load('results/SPM_training_results/training'+str(i_training)+'/episode'+str(start_episode)+'/checkpoint_actor_'+str(start_episode)+'.pth'))
    agent.actor_optimizer.load_state_dict(torch.load('results/SPM_training_results/training'+str(i_training)+'/episode'+str(start_episode)+'/checkpoint_actor_optimizer_'+str(start_episode)+'.pth'))
    agent.critic_local.load_state_dict(torch.load('results/SPM_training_results/training'+str(i_training)+'/episode'+str(start_episode)+'/checkpoint_critic_'+str(start_episode)+'.pth'))
    agent.critic_optimizer.load_state_dict(torch.load('results/SPM_training_results/training'+str(i_training)+'/episode'+str(start_episode)+'/checkpoint_critic_optimizer_'+str(start_episode)+'.pth'))

# call the function for training the agent
returns_list, checkpoints_list = ddpg(n_episodes=settings['number_of_training_episodes'], i_training=i_training, start_episode=start_episode)
total_returns_list_with_exploration.append(returns_list)
    

with open("results/SPM_training_results/total_returns_list_with_exploration.txt", "wb") as fp:   #Pickling, \\ -> / for mac.
   pickle.dump(total_returns_list_with_exploration, fp)

with open("results/SPM_training_results/checkpoints_list.txt", "wb") as fp:   #Pickling
   pickle.dump(checkpoints_list, fp)

ACTION = policy_heatmap(agent, T=25)








