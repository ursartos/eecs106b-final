"""
Setup of Project 2 python codebase
"""
from setuptools import setup

setup(name='proj2',
      version='0.1.0',
      description='EE 106B Project 2 code',
      package_dir = {'': 'src'},
      packages=['proj2'],
      install_requires=[],
      test_suite='test'
     )