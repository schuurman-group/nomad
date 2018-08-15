"""
Routines for running a TeraChem-Cloud computation with PyTC.
"""
import os
import numpy as np
import pytc
import nomad.core.glbl as glbl
import nomad.core.atom_lib as atom_lib
import nomad.core.trajectory as trajectory
import nomad.core.surface as surface
import nomad.integrals.centroid as centroid
import nomad.math.constants as constants


client = None


class PythonTC:
    """Object containing the PyTC properties and keys."""
    def __init__(self, host='http://fire-05-31', port=30080, engine='terachem'):
        # the PyTC client object
        self.TC = None

        # client options
        self.client_opts = dict(
            host = host,
            port = port,
            user = os.environ['TCCLOUD_USER'],
            api_key = os.environ['TCCLOUD_API_KEY'],
            engine = engine
                                )

        # job options
        self.job_opts = dict(
            atoms = glbl.crd_labels,
            charge = glbl.pytc['charge'],
            spinmult = glbl.pytc['spinmult'],
            closed_shell = glbl.pytc['closed_shell'],
            restricted = glbl.pytc['restricted'],
            method = glbl.pytc['guess_method'],
            basis = glbl.pytc['basis']
                             )

        # potential energy options
        self.fomo_opts = dict(
            casci = glbl.pytc['casci'],
            fon = glbl.pytc['fon'],
            closed = glbl.pytc['closed'],
            active = glbl.pytc['active'],
            cassinglets = glbl.pytc['cassinglets'],
            bond_order = glbl.pytc['bond_order']
                              )

        self.nstates = glbl.properties['n_states']

        # gradient options
        self.grad_opts = np.empty(self.nstates, dtype=object)

        # coupling options
        self.coup_opts = np.empty((self.nstates, self.nstates), dtype=object)

        # setup options for all pairs of states
        self._update_gradopts()

    def setup_client(self):
        """Returns the PyTC client and job."""
        self.TC = pytc.Client(**self.client_opts)
        self.TC.job_spec(**self.job_opts)

    def compute_sp(self, geom, state1=0, state2=0, name='E', jtype='energy', **opts):
        """Returns a dictionary from a single point calculation with
        extra options."""
        sp_opts = self.fomo_opts.copy()
        if jtype == 'gradient':
            sp_opts.update(self.grad_opts[state1])
        elif jtype == 'coupling':
            sp_opts.update(self.coup_opts[state1,state2])
        sp_opts.update(opts)
        results = self.TC.compute(name=name, job_type=jtype, geoms=geom,
                                  **opts)
        key = list(results.keys())[0]
        return results[key]

    def compute_energies(self, geom, name='E'):
        """Returns potential energies for the given molecular geometry."""
        return self.compute_sp(geom, name=name)['energy']

    def compute_gradient(self, geom, state, name='Grad'):
        """Returns potential energy gradients for the given geometry
        and state."""
        grad = self.compute_sp(geom, state1=state1, name=name+str(state),
                               jtype='gradient')['gradient']
        return grad.astype(float)

    def compute_coupling(self, geom, state1, state2, name='Coup'):
        """Returns nonadiabatic coupling for the given geometry
        and pair of states."""
        if state1 == state2:
            return np.zeros(np.geom.size)
        else:
            nac = self.compute_sp(geom, state1=state1, state2=state2,
                                  name=name+str(state1)+str(state2),
                                  jtype='coupling')['nacme']
            return nac.astype(float)

    def _update_gradopts(self):
        """Updates gradient options for the number of requested states."""
        for i in range(self.nstates):
            self.grad_opts[i] = dict(
                castarget = i,
                castargetmult = glbl.pytc['spinmult']
                                     )
            for j in range(self.nstates):
                self.coup_opts[i,j] = dict(
                    nacstate1 = i,
                    nacstate2 = j,
                    castargetmult = glbl.pytc['spinmult']
                                           )


def init_interface():
    """Initializes the PyTC calculation."""
    global client

    client = PythonTC()
    client.setup_client()


def evaluate_trajectory(traj, t=None):
    """Computes energy and computes all couplings."""
    state = traj.state
    nstates = traj.nstates

    tc_surf = surface.Surface()
    col_surf.add_data('geom', traj.x())

    props = client.compute_sp(traj.x(), state1=state, jtype='gradient')

    #tc_surf.add_data('mo', pack_mocoef())

    tc_surf.add_data('potential', props['energy'] + glbl.properties['pot_shift'])
    tc_surf.add_data('atom_pop', -props['charges']) # not exactly
    #tc_surf.add_data('sec_mom', sec_moms)

    # compute gradient on current state
    deriv = np.zeros((traj.dim, nstates, nstates))
    dipoles = np.zeros((3, nstates, nstates))
    deriv[:,state,state] = props['gradient']

    # run coupling to other states
    for i in range(nstates):
        #props = client.compute_sp(traj.x(), state1=i, jtype='gradient')
        #dipoles[:,i,i] = props['dipole_vector']
        if i != state:
            state_i = min(i,state)
            state_j = max(i,state)
            nad_coup = client.compute_coupling(traj.x(), state_i, state_j)
            deriv[:,state_i,state_j] =  nad_coup
            deriv[:,state_j,state_i] = -nad_coup

    tc_surf.add_data('derivative', deriv)
    tc_surf.add_data('dipole', dipoles)

    return tc_surf


def evaluate_centroid(cent, t=None):
    """Evaluates all requested electronic structure information at a
    centroid."""
    state = traj.state
    nstates = traj.nstates

    state_i = min(cent.states)
    state_j = max(cent.states)

    tc_surf = surface.Surface()
    col_surf.add_data('geom', cent.x())

    props = client.compute_sp(traj.x())

    #tc_surf.add_data('mo', pack_mocoef())

    tc_surf.add_data('potential', props['energy'] + glbl.properties['pot_shift'])
    tc_surf.add_data('atom_pop', -props['charges']) # not exactly

    deriv = np.zeros((cent.dim, nstates, nstates))
    if state_i != state_j:
        # run coupling between states
        nad_coup = client.compute_coupling(traj.x(), state_i, state_j)
        deriv[:,state_i,state_j] =  nad_coup
        deriv[:,state_j,state_i] = -nad_coup

    tc_surf.add_data('derivative', deriv)

    return tc_surf


def evaluate_coupling(traj):
    """Evaluates coupling between electronic states"""
    nstates = traj.nstates
    state = traj.state

    coup = np.zeros((nstates, nstates))
    vel  = traj.velocity()
    for i in range(nstates):
        if i != state:
            coup[state,i] = np.dot(vel, traj.derivative(state,i))
            coup[i,state] = -coup[state,i]
    traj.pes.add_data('coupling', coup)
