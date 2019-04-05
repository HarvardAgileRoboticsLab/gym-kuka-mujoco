import os
from setuptools import setup, find_packages
from glob import glob

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(name='gym_kuka_mujoco',
      version='0.0.1',
      install_requires=required,
      packages=find_packages(),
      package_data={
          'gym_kuka_mujoco.envs.assets': '*.xml',
          'gym_kuka_mujoco.envs.assets.kuka': '*.xml',
          'gym_kuka_mujoco.envs.assets.peg': '*.xml',
          'gym_kuka_mujoco.envs.assets.meshes': '*.stl',
          'gym_kuka_mujoco.envs.assets.hammer': '*.xml',
          'gym_kuka_mujoco.envs.assets.hole': '*.xml',
          'gym_kuka_mujoco.envs.assets.pushing': '*.xml',
          'gym_kuka_mujoco.envs.assets.valve': '*.xml',
      }
)
