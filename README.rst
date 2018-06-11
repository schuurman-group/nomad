nomad
=====
Nonadiabatic Multistate Adaptive Dynamics in Python

Created Nov. 11, 2015 -- M.S. Schuurman

Requirements
------------
Requires at least Python 3.3, NumPy v1.7.0, SciPy v0.12.0 and MPI4Py v2.0.0

Setup
-----
For faster performance, compile nuclear_gaussian.pyx and nuclear_dirac.pyx from the main directory with:
```
python setup.py build_ext --inplace
```
