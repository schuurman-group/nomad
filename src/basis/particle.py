import numpy as np
import gaussian
class particle:
    def __init__(self,dim,pid):
        # dimension of the particle (=3 for atom in cartesian coords)
        self.dim   = dim
        # particle identifier (integer, specifying atom number)
        self.pid   = pid 
        # create a particle with zero position and momentum
        self.x     = np.zeros(self.dim)
        self.p     = np.zeros(self.dim)
        # width/uncertainty
        self.width = 0.
        # mass of particle
        self.mass  = 0.
        # atomic number, if relevant
        self.atomic_num = 0
        # charge on the particle (i.e. Mulliken, etc.)
        self.charge = 0.
        # a string name for particle (i.e. atom number, mode number, etc.)
        self.name   = ''

#-----------------------------------------------------------------------------
#
# Integral Routines
#
#-----------------------------------------------------------------------------
    #
    # overlap of two particles
    #
    def overlap(self,other):
         S = complex(1.,0.)
         for i in range(self.dim):
             S = S * gaussian.overlap(self.x[i],self.p[i],self.width,
                                     other.x[i],other.p[i],other.width)
         return S

    #
    # del/dp matrix element between two particles
    #
    def deldp(self,other):
         dpval =  np.zeros(self.dim)
         for i in range(self.dim):
             dpval[i] = gaussian.deldp(self.x[i],self.p[i],self.width,
                                      other.x[i],other.p[i],other.width)
         return dpval * overlap(self,other)

    #
    # del/dx matrix element between two particles
    #
    def deldx(self,other):
         dxval =  np.zeros(self.dim)
         for i in range(self.dim):
             dxval[i] = gaussian.deldx(self.x[i],self.p[i],self.width,
                                      other.x[i],other.p[i],other.width)
         return dxval * overlap(self,other)

    #
    # del^2/dx^2 matrix element between two particles
    #
    def deld2x(self,other):
         d2xval = complex(0.,0.)
         for i in range(self.dim):
             d2xval = d2xval + gaussian.deld2x(self.x[i],self.p[i],self.width,
                                              other.x[i],other.p[i],other.width)
         return d2xval * overlap(self,other)

    #------------------------------------------------------------------------
    #
    # Routines to write/read info from text file
    #
    #------------------------------------------------------------------------
    #
    # write particle to file stream chkpt
    #
    def write_particle(self,chkpt):
        chkpt.write('{:2s}              particle name'.format(self.name))
        chkpt.write('{:10d}             particle ID'.format(self.pid))
        chkpt.write('{:10d}             atomic number'.format(self.atomic_num))
        chkpt.write('{:10d}             dimension'.format(self.dim))
        chkpt.write('{:16.10e}          width'.format(self.width))
        chkpt.write('{:16.10e}          mass'.format(self.mass))
        chkpt.write('{:16.10e}          charge'.format(self.charge))

    #
    # Reads particle written to file by write_particle
    #
    def read_particle(self,chkpt):
        self.name       = chkpt.readline()[0]
        self.pid        = int(chkpt.readline()[0])
        self.atomic_num = int(chkpt.readline()[0])
        self.dim        = int(chkpt.readline()[0])
        self.width      = float(chkpt.readline()[0])
        self.mass       = float(chkpt.readline()[0])
        self.charge     = float(chkpt.readline()[0])

