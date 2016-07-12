from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules=[
    Extension("gaussian", sources=["gaussian.pyx"], libraries=["m"])
]

setup(
  name = "Gaussian",
  ext_modules = cythonize(ext_modules)
)
