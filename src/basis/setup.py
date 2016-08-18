"""
Setup script for Cython basis modules.
"""
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules=[
    Extension('gaussian', sources=['gaussian.pyx'], libraries=['m']),
    Extension('dirac_delta', sources=['dirac_delta.pyx'], libraries=['m'])
]

setup(
  name = 'FMSpy',
  ext_modules = cythonize(ext_modules)
)
