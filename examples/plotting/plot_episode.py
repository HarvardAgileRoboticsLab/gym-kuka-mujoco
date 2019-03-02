#!/usr/bin/env /usr/local/bin/python3
import os
import argparse
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from stable_baselines.results_plotter import window_func, load_results, ts2xy, X_TIMESTEPS, X_EPISODES, X_WALLTIME

# Add the parent folder to the python path for imports.
import sys, os
sys.path.insert(0, os.path.abspath('..'))
from experiment_files import get_latest_experiment_dir

def plot_data(data, x_axis, y_axis, window=1, max_idx=-1, bounds="percentile", clip_low=np.inf, clip_high=np.inf, bounds_scale=1.0, **kwargs):
    if x_axis == X_TIMESTEPS:
        x_data = np.cumsum(data.l.values)
    elif x_axis == X_EPISODES:
        x_data = np.arange(len(data))
    elif x_axis == X_WALLTIME:
        x_data = data.t/3600.
    else:
        raise NotImplementedError
    
    y_data = data[y_axis].values

    x_data = x_data[:max_idx]
    y_data = y_data[:max_idx]

    def percentile_20(array, **kwargs):
        return np.percentile(array, 20, **kwargs)

    def percentile_80(array, **kwargs):
        return np.percentile(array, 80, **kwargs)

    # x_data = x_data[::10]
    # y_data = y_data[::10]

    # Smooth the data
    print('here 1')
    x_trimmed, y_mean = window_func(x_data, y_data, window, np.mean)
    print('here 2')

    

    # import pdb; pdb.set_trace()

    if bounds == "percentile":
        _, y_low = window_func(x_data, y_data, window, percentile_20)
        _, y_high = window_func(x_data, y_data, window, percentile_80)
        
        print('here 3')
        y_low = np.minimum(y_low, y_mean)
        y_high = np.maximum(y_high, y_mean)
        print('here 4')
        # plt.fill_between(x_trimmed, y_low, y_high, alpha=0.2)
    elif bounds == "std":
        _, y_std = window_func(x_data, y_data, window, np.std)
        y_std *= bounds_scale

        print('here 3.1')
        y_low = np.clip(y_mean-y_std, clip_low, clip_high)
        y_high = np.clip(y_mean+y_std, clip_low, clip_high)
        print('here 4.1')
    
    print('here 5')
    plt.fill_between(x_trimmed, y_low, y_high, alpha=0.2)
    plt.plot(x_trimmed, y_mean, **kwargs)
if __name__ == '__main__':
    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'directory', type=str, help='The directory of the experiment.')

    args = parser.parse_args()

    # Load the model if it's availeble, otherwise that latest checkpoint.
    experiment_dir = get_latest_experiment_dir(args.directory)
    data = load_results(experiment_dir) 
    
    # time variables
    # plot_data(data, X_EPISODES, 'tip_distance', label="plot 1")
    plt.plot([1,2],[2,1])
    plt.ylabel("Tip Distance")
    plt.xlabel("Episodes")
    plt.legend()

    plt.savefig("test.png")