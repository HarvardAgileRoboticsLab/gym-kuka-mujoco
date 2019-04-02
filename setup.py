from setuptools import setup, find_packages

setup(name='gym_kuka_mujoco',
      version='0.0.1',
      install_requires=[
        'gym[mujoco]',
        'commentjson'
      ],  # And any other dependencies foo needs
      packages=find_packages()
)
