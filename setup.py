"""
Setup script for the nomad package.
"""
from setuptools import find_packages
from numpy.distutils.core import setup
from numpy.distutils.core import Extension


def readme():
    """Read in the README file."""
    with open('README.rst', 'r') as f:
        return f.read()


def requirements():
    """Returns the requirement list."""
    with open('requirements.txt', 'r') as f:
        return [line.strip() for line in f.readlines()]


ext_modules = [
    Extension('nomad.compiled.nuclear_gaussian',
              sources=['nomad/compiled/nuclear_gaussian.f90']),
    Extension('nomad.compiled.nuclear_gaussian_ccs',
              sources=['nomad/compiled/nuclear_gaussian_ccs.f90']),
    Extension('nomad.compiled.nuclear_dirac',
              sources=['nomad/compiled/nuclear_dirac.f90'])
               ]

setup(
    name='nomad',
    version='0.2',
    description='Nonadiabatic Multi-state Adaptive Dynamics',
    long_description=readme(),
    keywords='quantum molecular dynamics excited state nonadiabatic chemistry',
    url='https://github.com/mschuurman/nomad',
    author='Michael S. Schuurman',
    license='LGPL-3.0',
    packages=find_packages(),
    scripts=['bin/nomad_driver', 'bin/nomad_extract'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Topic :: Scientific/Engineering :: Chemistry'
                 ],
    install_requires=requirements(),
    ext_modules=ext_modules
      )
