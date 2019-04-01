#!/usr/bin/env /usr/local/bin/python3
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines.results_plotter import load_results, ts2xy, X_TIMESTEPS, X_EPISODES, X_WALLTIME

# Add the parent folder to the python path for imports.
from plot_episode import plot_data
import sys, os
sys.path.insert(0, os.path.abspath('..'))
from experiment_files import get_latest_experiment_dir


# Load the PPO data.
# Torque
directory = os.path.join(os.environ['HOME'],'Google Drive','DriveSync','DirectTorqueController_HammerEnv_PPO2_0')
directory = os.path.abspath(directory)
experiment_dir = get_latest_experiment_dir(directory)
torque_data = load_results(experiment_dir) 
print(experiment_dir)

# PD
directory = os.path.join(os.environ['HOME'],'Google Drive','DriveSync','RelativePDController_HammerEnv_PPO2_0')
directory = os.path.abspath(directory)
experiment_dir = get_latest_experiment_dir(directory)
print(experiment_dir)
pd_data = load_results(experiment_dir) 

# ID
directory = os.path.join(os.environ['HOME'],'Google Drive','DriveSync','RelativeInverseDynamicsController_HammerEnv_PPO2_0')
directory = os.path.abspath(directory)
experiment_dir = get_latest_experiment_dir(directory)
print(experiment_dir)
id_data = load_results(experiment_dir) 

# Impedance
directory = os.path.join(os.environ['HOME'],'Google Drive','DriveSync','ImpedanceControllerV2_HammerEnv_PPO2_0')
directory = os.path.abspath(directory)
experiment_dir = get_latest_experiment_dir(directory)
print(experiment_dir)
impedance_data = load_results(experiment_dir) 

# Plot the PPO data.
plt.figure()
plot_data(torque_data, X_EPISODES, 'nail_depth', window=500, max_idx=20000, label="torque")
plot_data(pd_data, X_EPISODES, 'nail_depth', window=500, max_idx=20000, label="PD")
plot_data(id_data, X_EPISODES, 'nail_depth', window=500, max_idx=20000 ,label="ID")
plot_data(impedance_data, X_EPISODES, 'nail_depth', window=500, max_idx=20000, label="impedance")
plt.legend(fontsize=16)
plt.xlabel('Episodes', fontsize=16)
plt.ylabel('Nail Depth (m)', fontsize=16)
plt.title('PPO', fontsize=20)

# Load the SAC data.
# Torque
directory = os.path.join(os.environ['HOME'],'Google Drive','DriveSync','DirectTorqueController_HammerEnv_SAC_0')
directory = os.path.abspath(directory)
experiment_dir = get_latest_experiment_dir(directory)
torque_data = load_results(experiment_dir) 
print(experiment_dir)

# PD
directory = os.path.join(os.environ['HOME'],'Google Drive','DriveSync','RelativePDController_HammerEnv_SAC_0')
directory = os.path.abspath(directory)
experiment_dir = get_latest_experiment_dir(directory)
print(experiment_dir)
pd_data = load_results(experiment_dir) 

# ID
directory = os.path.join(os.environ['HOME'],'Google Drive','DriveSync','RelativeInverseDynamicsController_HammerEnv_SAC_0')
directory = os.path.abspath(directory)
experiment_dir = get_latest_experiment_dir(directory)
print(experiment_dir)
id_data = load_results(experiment_dir) 

# Impedance
directory = os.path.join(os.environ['HOME'],'Google Drive','DriveSync','ImpedanceControllerV2_HammerEnv_SAC_0')
directory = os.path.abspath(directory)
experiment_dir = get_latest_experiment_dir(directory)
print(experiment_dir)
impedance_data = load_results(experiment_dir) 

# Plot the PPO data.
plt.figure()
plot_data(torque_data, X_EPISODES, 'nail_depth', window=250, max_idx=5000, label="torque")
plot_data(pd_data, X_EPISODES, 'nail_depth', window=250, max_idx=5000, label="PD")
plot_data(id_data, X_EPISODES, 'nail_depth', window=250, max_idx=5000 ,label="ID")
plot_data(impedance_data, X_EPISODES, 'nail_depth', window=100, max_idx=5000, label="impedance")
# plt.legend(fontsize=24)
plt.xlabel('Episodes', fontsize=24)
plt.ylabel('Nail Depth (m)', fontsize=24)
plt.title('SAC', fontsize=30)
plt.show()
# import pdb; pdb.set_trace()