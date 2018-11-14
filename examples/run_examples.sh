#!/bin/bash

# Train 10 random restarts
for i in {1..10}
do
  python run_baselines.py --alg=ppo2 --env=PegInsertion-v0 --network=mlp --num_timesteps=1e7 --save_path=~/models/PegInsertion --play --log_interval=1 --gamma=1 --lr=1e-3 --save_interval=10 --num_env=8
done
