"""
Setup script for Cython basis modules.
"""
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules=[
    Extension('nuclear_gaussian', sources=['nuclear_gaussian.pyx'], libraries=['m']),
    Extension('nuclear_dirac', sources=['nuclear_dirac.pyx'], libraries=['m'])
]

setup(
  name = 'FMSpy',
  ext_modules = cythonize(ext_modules)
)
