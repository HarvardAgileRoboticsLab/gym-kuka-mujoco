import os
import sys

import cloudpickle
from gym_kuka_mujoco.utils.load_model import load_params, load_model

sys.path.insert(0, os.path.abspath('..'))
from experiment_files import get_latest_experiment_dir, get_params, get_model, get_latest_checkpoint

# Get the model path.
log_dir = os.path.join(os.environ['OPENAI_LOGDIR'], 'stable', '2019-04-01', "18:45:39.122285")
log_dir = get_latest_experiment_dir(log_dir)
params_path = get_params(log_dir)
model_path = get_model(log_dir)
if model_path is None:
    model_path = get_latest_checkpoint(log_dir)

# Load the model with stable-baselines.
params = load_params(params_path)
env, model = load_model(model_path, params)

# Load the model with tensorflow.
with open(model_path, "rb") as file:
    data, params = cloudpickle.load(file)

for _ in range(10):
    obs = env.reset()
    action, _ = model.predict(obs, deterministic=True)
    print(action)