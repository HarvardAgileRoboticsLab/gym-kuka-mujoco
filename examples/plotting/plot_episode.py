#!/usr/bin/env /usr/local/bin/python3
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines.results_plotter import window_func, load_results, ts2xy, X_TIMESTEPS, X_EPISODES, X_WALLTIME

# Add the parent folder to the python path for imports.
import sys, os
sys.path.insert(0, os.path.abspath('..'))
from experiment_files import get_latest_experiment_dir

def plot_data(data, x_axis, y_axis, window=1, max_idx=-1, **kwargs):
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

    # Smooth the data
    x_trimmed, y_mean = window_func(x_data, y_data, window, np.mean)
    # _, y_std = window_func(x_data, y_data, window, np.std)

    _, y_low = window_func(x_data, y_data, window, percentile_20)
    _, y_high = window_func(x_data, y_data, window, percentile_80)

    y_low = np.minimum(y_low, y_mean)
    y_high = np.maximum(y_high, y_mean)
    plt.plot(x_trimmed, y_mean, **kwargs)
    # plt.fill_between(x_trimmed, y_mean-y_std, y_mean+y_std, alpha=0.2)
    plt.fill_between(x_trimmed, y_low, y_high, alpha=0.2)

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

    plt.show()