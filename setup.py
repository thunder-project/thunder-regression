#!/usr/bin/env python

from setuptools import setup

version = '1.0.0'

required = open('requirements.txt').read().split('\n')

setup(
    name='thunder-regression',
    version=version,
    description='algorithms for mass univariate regression',
    author='jwittenbach',
    author_email='the.freeman.lab@gmail.com',
    url='https://github.com/freeman-lab/thunder-regression',
    packages=['regression'],
    install_requires=required,
    long_description='See ' + 'https://github.com/freeman-lab/thunder-regression',
    license='MIT'
)
