import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines.results_plotter import load_results, ts2xy, X_TIMESTEPS, X_EPISODES, X_WALLTIME
from experiment_files import get_latest_experiment_dir

def plot_data(data, x_axis, y_axis, **kwargs):
    if x_axis == X_TIMESTEPS:
        x_data = np.cumsum(data.l.values)
    elif x_axis == X_EPISODES:
        x_data = np.arange(len(data))
    elif x_axis == X_WALLTIME:
        x_data = data.t/3600.
    else:
        raise NotImplementedError
    
    y_data = data[y_axis].values
    plt.plot(x_data, y_data, **kwargs)

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
    plot_data(data, X_EPISODES, 'tip_distance', label="plot 1")
    plt.ylabel("Tip Distance")
    plt.xlabel("Episodes")
    plt.legend()

    plt.show()