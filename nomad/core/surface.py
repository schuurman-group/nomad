"""
Routines for handling the potential energy surface.

All calls to update the pes are localized here.  This facilitates parallel
execution of potential evaluations which is essential for ab initio PES.
"""
import copy
import numpy as np
import nomad.math.constants as constants
import nomad.core.glbl as glbl


pes_cache  = dict()


class Surface:
    """Object containing potential energy surface data."""
    def __init__(self):
        self.standard_objs   = ['geom','potential','derivative','hessian','coupling']
        self.mo_objs         = ['mo']
        self.electronic_objs = ['dipole','atom_pop','sec_mom']
        self.vibronic_objs   = ['diabat_pot','diabat_deriv','diabat_hessian',
                                'adt_mat','dat_mat','nac','scalar_coup']
        self.all_objs      = (self.standard_objs + self.mo_objs +
                              self.electronic_objs + self.vibronic_objs)

        # these are the standard quantities ALL interface_data objects return
        self.data = dict()

    def rm_data(self, key):
        """Adds new item to dictionary"""
        del self.data[key]

    def add_data(self, key, value):
        """Adds new item to dictionary"""
        if key in self.all_objs:
            self.data[key] = value
        else:
            raise KeyError('Cannot add key='+str(key)+' to Surface instance: invalid key')

    def get_data(self, key):
        """Adds new item to dictionary"""
        if key in self.data:
            return self.data[key]
        else:
            raise ValueError('(get_data('+str(key)+') from Surface: datum not present')

    def avail_data(self):
        """Adds new item to dictionary"""
        return self.data.keys()

    def valid_data(self, key):
        """Return true if data is valid for addition to surface object"""
        return key in self.all_objs

    def copy(self):
        """Creates a copy of a Surface object."""
        new_surface = Surface()

        # required potential data
        for key,value in self.data.items():
            new_surface.data[key] = copy.deepcopy(value)

        return new_surface


def update_pes(wfn, update_integrals=True):
    """Updates the potential energy surface."""
    global pes_cache
    success = True

    # this conditional checks to see if we actually need centroids,
    # even if propagator requests them
    if update_integrals:
        glbl.modules['integrals'].update(wfn)

    if glbl.mpi['parallel']:
        # update electronic structure
        exec_list = []
        n_total = 0 # this ensures traj.0 is on proc 0, etc.

        for i in range(wfn.n_traj()):
            if wfn.traj[i].active and not cached(wfn.traj[i].label,
                                                    wfn.traj[i].x()):

                n_total += 1
                if n_total % glbl.mpi['nproc'] == glbl.mpi['rank']:
                    exec_list.append(['traj',wfn.traj[i]])

        if update_integrals and glbl.modules['integrals'].require_centroids:
            # now update electronic structure in a controled way to allow for
            # parallelization
            for i in range(wfn.n_traj()):
                for j in range(i):
                    if glbl.modules['integrals'].centroid_required[i][j] and not \
                                             cached(glbl.modules['integrals'].centroids[i][j].label,
                                                    glbl.modules['integrals'].centroids[i][j].x()):
                        n_total += 1
                        if n_total % glbl.mpi['nproc'] == glbl.mpi['rank']:
                            exec_list.append(['cent',glbl.modules['integrals'].centroids[i][j]])

        local_results = []

        print("rank="+str(glbl.mpi['rank'])+", length exec_lst="+str(len(exec_list)))
        for i in range(len(exec_list)):
            if exec_list[i][0] is 'traj':
                pes_calc = glbl.modules['interface'].evaluate_trajectory(exec_list[i][1], t=wfn.time)
            elif exec_list[i][0] is 'cent':
                pes_calc = glbl.modules['interface'].evaluate_centroid(exec_list[i][1], t=wfn.time)
            else:
                raise TypeError('type='+str(type(exec_list[i]))+
                                'not recognized')
                pes_calc = None
            local_results.append([exec_list[i][1].label,pes_calc])

        global_results = glbl.mpi['comm'].allgather(local_results)

        # update the cache
        for i in range(glbl.mpi['nproc']):
            for j in range(len(global_results[i])):
                pes_cache[global_results[i][j][0]] = global_results[i][j][1]
        # update the bundle:
        # live trajectories
        for i in range(wfn.n_traj()):
            if wfn.traj[i].alive:
                wfn.traj[i].update_pes_info(pes_cache[wfn.traj[i].label])

        # and centroids
        if update_integrals and glbl.modules['integrals'].require_centroids:
            for i in range(wfn.n_traj()):
                for j in range(i):
                    c_label = glbl.modules['integrals'].centroids[i][j].label
                    if c_label in pes_cache:
                        glbl.modules['integrals'].centroids[i][j].update_pes_info(pes_cache[c_label])
                        glbl.modules['integrals'].centroids[j][i] = glbl.modules['integrals'].centroids[i][j]

    # if parallel overhead not worth the time and effort (eg. pes known in closed form),
    # simply run over trajectories in serial (in theory, this too could be cythonized,
    # but unlikely to ever be bottleneck)
    else:

        # iterate over trajectories..
        for i in range(wfn.n_traj()):
            if wfn.traj[i].active:
                if not cached(wfn.traj[i].label, wfn.traj[i].x()):
                    pes_traji = glbl.modules['interface'].evaluate_trajectory(wfn.traj[i], wfn.time)
                else:
                    pes_traji = pes_cache[wfn.traj[i].label]
                wfn.traj[i].update_pes_info(pes_traji)
                pes_cache[wfn.traj[i].label] = pes_traji.copy()
        # ...and centroids if need be
        if update_integrals and glbl.modules['integrals'].require_centroids:
            for i in range(wfn.n_traj()):
                for j in range(i):
                # if centroid not initialized, skip it
                    if glbl.modules['integrals'].centroid_required[i][j]:
                        if not cached(glbl.modules['integrals'].centroids[i][j].label,
                                      glbl.modules['integrals'].centroids[i][j].x()):
                            pes_centij = glbl.modules['interface'].evaluate_centroid(
                                         glbl.modules['integrals'].centroids[i][j], wfn.time)
                        else:
                            pes_centij = pes_cache[glbl.modules['integrals'].centroids[i][j].label]
                        glbl.modules['integrals'].centroids[i][j].update_pes_info(pes_centij)
                        glbl.modules['integrals'].centroids[j][i] = glbl.modules['integrals'].centroids[i][j]
                        pes_cache[glbl.modules['integrals'].centroids[i][j].label] = pes_centij

    return success


def update_pes_traj(traj):
    """Updates a single trajectory

    Used during spawning.
    """
    results = None

    if glbl.mpi['rank'] == 0:
        results = glbl.modules['interface'].evaluate_trajectory(traj)

    if glbl.mpi['parallel']:
        results = glbl.mpi['comm'].bcast(results, root=0)
        glbl.mpi['comm'].barrier()

    traj.update_pes_info(results)


def cached(label, geom):
    """Returns True if the surface in the cache corresponds to the current
    trajectory (don't recompute the surface)."""
    global pes_cache

    if label not in pes_cache:
        return False

    dg = np.linalg.norm(geom - pes_cache[label].get_data('geom'))
    if dg <= constants.fpzero:
        return True

    return False
