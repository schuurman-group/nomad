nomad
=====
Nonadiabatic Multistate Adaptive Dynamics

Created Nov. 11, 2015 -- M.S. Schuurman

Requirements
------------
Requires at least Python 3.3, NumPy v1.7.0, SciPy v0.12.0, cython-0.28.3, and MPI4Py v2.0.0.
The `Anaconda package distrubution <https://anaconda.org/>`_ is suggested.

Additionally, you need the openmpi headers.

```bash
sudo apt-get install libopenmpi-dev
```

Installation
------------
To create a local nomad directory and compile, use::

    $ git clone https://github.com/mschuurman/nomad.git
    $ cd nomad
    $ python setup.py install

This will also install the nomad driver (nomad.py) to the path.
