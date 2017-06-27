"""
Setup script for Cython basis modules.
"""
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize


def readme():
    with open('README.md', 'r') as f:
        return f.read()


ext_modules=[
    Extension('src.integrals.nuclear_gaussian',
              sources=['src/integrals/nuclear_gaussian.pyx'], libraries=['m']),
    Extension('src.integrals.nuclear_dirac',
              sources=['src/integrals/nuclear_dirac.pyx'], libraries=['m'])
]

setup(
    name='FMSpy',
    version='0.1',
    description='Full multiple spawning molecular dynamics in Python',
    long_description=readme(),
    keywords='quantum molecular dynamics excited state nonadiabatic chemistry',
    url='https://github.com/mschuurman/FMSpy',
    author='Michael S. Schuurman',
    license='LGPL-3.0',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Topic :: Scientific/Engineering :: Chemistry'
                 ],
    install_requires=['numpy>=1.7.0', 'scipy>=0.12.0', 'mpi4py>=2.0.0'],
    ext_modules = cythonize(ext_modules)
      )
