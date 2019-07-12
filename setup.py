#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='fpcal',
      version='0.1',
      description='Errors-in Bayesian focal plane calibration',
      author='Caleb Marshall',
      author_email='blah@nofuckingway.hotmail',
      url='https://github.com/dubiousbreakfast/fpcal',
      packages=find_packages(),
      install_requires=['pandas <= 24.0', 'matplotlib < 3.0',
                        'corner', 'emcee>2.2.0', 'seaborn']
     )

