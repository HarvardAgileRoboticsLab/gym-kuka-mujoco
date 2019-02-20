import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines.results_plotter import load_results, ts2xy, X_TIMESTEPS, X_EPISODES, X_WALLTIME
from experiment_files import get_latest_experiment_dir

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
    n_timesteps = np.cumsum(data.l.values)
    n_episodes = np.arange(len(data))
    wall_time = data.t/3600.

    # plot
    plt.plot(n_timesteps, data.r.values)
    plt.show()