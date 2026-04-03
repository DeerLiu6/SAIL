from setuptools import find_packages
from distutils.core import setup

setup(name='SAIL',
      version='1.0.0',
      author='Deer Liu',
      license="BSD-3-Clause",
      packages=find_packages(),
      description='RL environments for G1 Robots',
      install_requires=[
          'isaacgym', 
          'rsl-rl', 
          'matplotlib', 
          'numpy==1.23.5', 
          'tensorboard', 
          'mujoco==3.2.3', 
          'pyyaml', 
          'pydelatin', 
          'pyfqmr',
      ])
