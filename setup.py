from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(name='gym_kuka_mujoco',
      version='0.0.1',
      install_requires=required,  # And any other dependencies foo needs
      packages=find_packages()
)
