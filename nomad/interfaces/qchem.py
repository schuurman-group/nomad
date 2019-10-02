"""
Routines for running a Columbus computation.
"""
import sys
import os
import shutil
import numpy as np
import nomad.math.constants as constants
import nomad.core.glbl as glbl
import nomad.core.atom_lib as atom_lib
import nomad.core.trajectory as trajectory
import nomad.core.surface as surface
import nomad.integrals.centroid as centroid

#----------------------------------------------------------------
#
# Functions called from interface object
#
#----------------------------------------------------------------
def init_interface():
    """Initializes the Columbus calculation from the Columbus input."""

    # setup working directories
    # input and restart are shared
    input_path    = glbl.paths['cwd']+'/input'
    restart_path  = glbl.paths['cwd']+'/restart'
    # ...but each process has it's own work directory
    work_path     = glbl.paths['cwd']+'/work.'+str(glbl.mpi['rank'])

    # set atomic symbol, number, mass,
    natm    = len(glbl.properties['crd_labels']) // p_dim
    a_sym   = glbl.properties['crd_labels'][::p_dim]

    a_data  = []
    # we need to go through this to pull out the atomic numbers for
    # correct writing of input
    for i in range(natm):
        if atom_lib.valid_atom(a_sym[i]):
            a_data.append(atom_lib.atom_data(a_sym[i]))
        else:
            raise ValueError('Atom: '+str(a_sym[i])+' not found in library')

    # masses are au -- columbus geom reads mass in amu
    a_mass  = [a_data[i][1]/constants.amu2au for i in range(natm)]
    a_num   = [a_data[i][2] for i in range(natm)]

    if os.path.exists(work_path):
        shutil.rmtree(work_path)
    os.makedirs(work_path)

    if glbl.mpi['rank'] == 0:
        if os.path.exists(restart_path):
            shutil.rmtree(restart_path)
        os.makedirs(restart_path)

    # make sure process 0 is finished populating the input directory
    if glbl.mpi['parallel']:
        glbl.mpi['comm'].barrier()


def evaluate_trajectory(traj, t=None):

    label   = traj.label
    state   = traj.state
    nstates = traj.nstates

    if label < 0:
        print('evaluate_trajectory called with ' +
              'id associated with centroid, label=' + str(label))

    # create surface object to hold potential information
    qchem_surf = surface.Surface()
    qchem_surf.add_data('geom', traj.x())

    return qchem_surf


def evaluate_centroid(cent, t=None):
    """Evaluates all requested electronic structure information at a
    centroid."""
    global n_cart

    label   = cent.label
    nstates = cent.nstates

    if label >= 0:
        print('evaluate_centroid called with ' +
              'id associated with trajectory, label=' + str(label))

    state_i = min(cent.states)
    state_j = max(cent.states)

    # create surface object to hold potential information
    qchem_surf = surface.Surface()
    qchem_surf.add_data('geom', cent.x())

    return qchem_surf


def evaluate_coupling(traj):
    """evaluate coupling between electronic states"""
    nstates = traj.nstates
    state   = traj.state

    # effective coupling is the nad projected onto velocity
    coup = np.zeros((nstates, nstates))
    vel  = traj.velocity()
    for i in range(nstates):
        if i != state:
            coup[state,i] = np.dot(vel, traj.derivative(state,i))
            coup[i,state] = -coup[state,i]
    traj.pes.add_data('coupling', coup)



def append_log(label, listing_file, time):
    """Grabs key output from columbus listing files.

    Useful for diagnosing electronic structure problems.
    """
    # check to see if time is given, if not -- this is a spawning
    # situation
    if time is None:
        tstr = 'spawning'
    else:
        tstr = str(time)

    # open the running log for this process
    #log_file = open(glbl.home_path+'/columbus.log.'+str(glbl.mpi['rank']), 'a')
    log_file = open('qchem.log.'+str(glbl.mpi['rank']), 'a')


    log_file.close()


def write_qchem_geom(geom):
    """Writes a array of atoms to a COLUMBUS style geom file."""



