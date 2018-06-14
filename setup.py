"""
Setup script for the nomad package.
"""
from setuptools import setup
from setuptools.extension import Extension
from setuptools import find_packages
from Cython.Build import cythonize


def readme():
    """Read in the README file."""
    with open('README.rst', 'r') as f:
        return f.read()


ext_modules=[
    Extension('nomad.integrals.nuclear_gaussian',
              sources=['nomad/integrals/nuclear_gaussian.pyx'], libraries=['m']),
    Extension('nomad.integrals.nuclear_gaussian_ccs',
              sources=['nomad/integrals/nuclear_gaussian_ccs.pyx'], libraries=['m']),
    Extension('nomad.integrals.nuclear_dirac',
              sources=['nomad/integrals/nuclear_dirac.pyx'], libraries=['m'])
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
    install_requires=['numpy>=1.7.0', 'scipy>=0.12.0', 'h5py>=2.5.0',
                      'mpi4py>=2.0.0'],
    ext_modules = cythonize(ext_modules)
      )
