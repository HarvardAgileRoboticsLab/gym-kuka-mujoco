from setuptools import setup, find_packages
from pip.req import parse_requirements

# parse_requirements() returns generator of pip.req.InstallRequirement objects
install_reqs = parse_requirements('requirements.txt')

# reqs is a list of requirement
reqs = [str(ir.req) for ir in install_reqs]

setup(name='gym_kuka_mujoco',
      version='0.0.1',
      install_requires=reqs,  # And any other dependencies foo needs
      packages=find_packages()
)
