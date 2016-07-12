import sys
import re
import copy
import numpy as np
import src.fmsio.glbl as glbl
import src.basis.gaussian as gaussian
import src.vcham.hampar as ham

particle_name =  ['H','D','T','He','Li','Be','B','C','N','O','F','Ne',
                 'Na','Mg','Al','Si','P','S','Cl','Ar']
particle_width = [4.5, 4.5, 4.5, 0.0,0.0,0.0, 0.0, 22.5,19.5,13.0,8.5,0.0,
                  0.0, 0.0, 0.0, 0.0,0.0,17.5,0.0,0.0]
particle_mass  = [1.0,2.0,3.0,4.0, 7.0, 9.0,11.0,12.0,14.0,16.0,19.0,20.0,
                  23.0,24.0,27.0,28.0,31.0,32.0,35.45,40.0]
particle_anum  = [ 1., 1., 1., 2.,  3.,  4., 5., 6., 7., 8., 9., 10.,
                   11., 12., 13., 14., 15., 16., 17., 18.]

def valid_particle(particle):
    if(particle.name in particle_name):
        return True
    else:
        return False

def load_particle(particle):
    q = re.compile('q[0-9]',re.IGNORECASE)
    if particle.name in particle_name:
        index = particle_name.index(particle.name)
        particle.width = particle_width[index]
        particle.mass  = particle_mass[index]*glbl.mass2au
        particle.anum  = particle_anum[index]
        if(particle.width == 0.):
            outfile.write('WARNING: particle '+str(particle.name)+' in library, but width = 0')
    elif q.match(particle.name):
        if glbl.fms['interface'] == 'vibronic':
            particle.mass = 1.0/float(ham.freqmap[particle.name])
        else:
            particle.mass  = 1.

def create_particle(pid,dim,name,width,mass):
    new_particle       = particle(dim,pid)
    new_particle.name  = name
    new_particle.width = width
    new_particle.mass  = mass
    return new_particle

def copy_part(orig_part):
    new_part = particle(orig_part.dim,orig_part.pid)
    new_part.width  = copy.copy(orig_part.width)
    new_part.mass   = copy.copy(orig_part.mass)
    new_part.anum   = copy.copy(orig_part.anum)
    new_part.charge = copy.copy(orig_part.charge)
    new_part.name   = copy.copy(orig_part.name)
    new_part.x      = copy.deepcopy(orig_part.x)
    new_part.p      = copy.deepcopy(orig_part.p)
    return new_part

class particle:
    def __init__(self,dim,pid):
        # dimension of the particle (=3 for atom in cartesian coords)
        self.dim   = int(dim)
        # particle identifier (integer, specifying atom number)
        self.pid   = int(pid)
        # create a particle with zero position and momentum
        self.x     = np.zeros(self.dim,dtype=np.float)
        self.p     = np.zeros(self.dim,dtype=np.float)
        # width/uncertainty
        self.width = 0.
        # mass of particle
        self.mass  = 0.
        # atomic number, if relevant
        self.anum  = 0
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
         return np.fromiter((gaussian.deldp(self.x[i],self.p[i],self.width,
                                           other.x[i],other.p[i],other.width)
                                          for i in range(self.dim)),dtype=np.complex)


    #
    # del/dx matrix element between two particles
    #
    def deldx(self,other):
         return np.fromiter((gaussian.deldx(self.x[i],self.p[i],self.width,
                                          other.x[i],other.p[i],other.width)
                                          for i in range(self.dim)),dtype=np.complex)

    #
    # del^2/dx^2 matrix element between two particles
    #
    def deld2x(self,other):
#         print("x1,p1,w1,x2,p2,w2="+str(self.x[0])+" "+str(self.p[0])+" "+str(self.width)+" "+str(other.x[0])+" "+str(other.p[0])+" "+str(other.width)+" ")
         d2xval = np.fromiter((gaussian.deld2x(self.x[i],self.p[i],self.width,
                               other.x[i],other.p[i],other.width)
                               for i in range(self.dim)),dtype=np.complex)
#         print("d2xval = "+str(d2xval))
         return sum(d2xval)

    #------------------------------------------------------------------------
    #
    # Routines to write/read info from text file
    #
    #------------------------------------------------------------------------
    #
    # write particle to file stream chkpt
    #
    def write_particle(self,chkpt):
        chkpt.write('        {:2s}            particle name\n'.format(self.name))
        chkpt.write('{:10d}            particle ID\n'.format(self.pid))
        chkpt.write('{:10.1f}            atomic number\n'.format(self.anum))
        chkpt.write('{:10d}            dimension\n'.format(self.dim))
        chkpt.write('{:16.10e}      width\n'.format(self.width))
        chkpt.write('{:16.10e}      mass\n'.format(self.mass))
        chkpt.write('{:16.10e}      charge\n'.format(self.charge))

    #
    # Reads particle written to file by write_particle
    #
    def read_particle(self,chkpt):
        self.name       = str(chkpt.readline().split()[0])
        self.pid        = int(chkpt.readline().split()[0])
        self.atomic_num = float(chkpt.readline().split()[0])
        self.dim        = int(chkpt.readline().split()[0])
        self.width      = float(chkpt.readline().split()[0])
        self.mass       = float(chkpt.readline().split()[0])
        self.charge     = float(chkpt.readline().split()[0])
