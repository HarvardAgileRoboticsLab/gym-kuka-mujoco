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
directory = os.path.join(os.environ['HOME'],'Google Drive','DriveSync','DirectTorqueController_PegInsertionEnv_Hole25_new_PPO2_0')
directory = os.path.abspath(directory)
experiment_dir = get_latest_experiment_dir(directory)
torque_data = load_results(experiment_dir) 
print(experiment_dir)

# PD
directory = os.path.join(os.environ['HOME'],'Google Drive','DriveSync','RelativePDController_PegInsertionEnv_Hole25_new_PPO2_0')
directory = os.path.abspath(directory)
experiment_dir = get_latest_experiment_dir(directory)
print(experiment_dir)
pd_data = load_results(experiment_dir) 

# ID
directory = os.path.join(os.environ['HOME'],'Google Drive','DriveSync','RelativeInverseDynamicsController_PegInsertionEnv_Hole25_new_PPO2_0')
directory = os.path.abspath(directory)
experiment_dir = get_latest_experiment_dir(directory)
print(experiment_dir)
id_data = load_results(experiment_dir) 

# Impedance
directory = os.path.join(os.environ['HOME'],'Google Drive','DriveSync','ImpedanceControllerV2_PegInsertionEnv_Hole25_new_PPO2_0')
directory = os.path.abspath(directory)
experiment_dir = get_latest_experiment_dir(directory)
print(experiment_dir)
impedance_data = load_results(experiment_dir) 

# Plot the PPO data.
plt.figure()
plot_data(torque_data, X_EPISODES, 'success', window=500, max_idx=20000, label="torque", bounds="std", clip_low=0.0, clip_high=1.0, bounds_scale=0.5)
plot_data(pd_data, X_EPISODES, 'success', window=500, max_idx=20000, label="PD", bounds="std", clip_low=0.0, clip_high=1.0, bounds_scale=0.5)
plot_data(id_data, X_EPISODES, 'success', window=500, max_idx=20000 ,label="ID", bounds="std", clip_low=0.0, clip_high=1.0, bounds_scale=0.5)
plot_data(impedance_data, X_EPISODES, 'success', window=500, max_idx=20000, label="impedance", bounds="std", clip_low=0.0, clip_high=1.0, bounds_scale=0.5)
# plt.legend(fontsize=16)
plt.xlabel('Episodes', fontsize=24)
plt.ylabel('Success Rate', fontsize=24)
plt.title('Insertion (PPO)', fontsize=30)
plt.tight_layout()

# Load the SAC data.
# Torque
directory = os.path.join(os.environ['HOME'],'Google Drive','DriveSync','DirectTorqueController_PegInsertionEnv_Hole25_new_SAC_0')
directory = os.path.abspath(directory)
experiment_dir = get_latest_experiment_dir(directory)
torque_data = load_results(experiment_dir) 
print(experiment_dir)

# PD
directory = os.path.join(os.environ['HOME'],'Google Drive','DriveSync','RelativePDController_PegInsertionEnv_Hole25_new_SAC_0')
directory = os.path.abspath(directory)
experiment_dir = get_latest_experiment_dir(directory)
print(experiment_dir)
pd_data = load_results(experiment_dir) 

# ID
directory = os.path.join(os.environ['HOME'],'Google Drive','DriveSync','RelativeInverseDynamicsController_PegInsertionEnv_Hole25_new_SAC_0')
directory = os.path.abspath(directory)
experiment_dir = get_latest_experiment_dir(directory)
print(experiment_dir)
id_data = load_results(experiment_dir) 

# Impedance
directory = os.path.join(os.environ['HOME'],'Google Drive','DriveSync','ImpedanceControllerV2_PegInsertionEnv_Hole25_new_SAC_0')
directory = os.path.abspath(directory)
experiment_dir = get_latest_experiment_dir(directory)
print(experiment_dir)
impedance_data = load_results(experiment_dir) 

# Plot the PPO data.
plt.figure()
plot_data(torque_data, X_EPISODES, 'success', window=500, max_idx=30000, bounds="std", clip_low=0.0, clip_high=1.0, bounds_scale=0.5, label="torque")
plot_data(pd_data, X_EPISODES, 'success', window=500, max_idx=30000, bounds="std", clip_low=0.0, clip_high=1.0, bounds_scale=0.5, label="PD")
plot_data(id_data, X_EPISODES, 'success', window=500, max_idx=30000, bounds="std", clip_low=0.0, clip_high=1.0, bounds_scale=0.5, label="ID")
plot_data(impedance_data, X_EPISODES, 'success', window=500, max_idx=30000, bounds="std", clip_low=0.0, clip_high=1.0, bounds_scale=0.5, label="impedance")
# plt.legend(fontsize=24)
plt.xlabel('Episodes', fontsize=24)
plt.ylabel('Success Rate', fontsize=24)
plt.title('Insertion (SAC)', fontsize=30)
plt.tight_layout()
plt.show()
# import pdb; pdb.set_trace()