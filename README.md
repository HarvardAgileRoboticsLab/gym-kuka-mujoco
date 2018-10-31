# gym-kuka-mujoco
An OpenAI gym environment for the Kuka arm.

## Installation

### Just the Environment
```
pip install -e .
```
### The Environment and Baselines Code for Learning
```
pip install tensorflow
pip install -r requirements.txt
```

## Example
Train on the KukaEnv with the baselines PPO implementation using
```
python examples/run_baselines.py --alg=ppo2 --env=KukaMujoco-v0 --network=mlp --num_timesteps=2e7 --play
```