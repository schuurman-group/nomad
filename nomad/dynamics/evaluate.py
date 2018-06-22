"""
Routines for handling the potential energy surface.

All calls to update the pes are localized here.  This facilitates parallel
execution of potential evaluations which is essential for ab initio PES.
"""
import numpy as np
import nomad.utils.constants as constants
import nomad.parse.glbl as glbl
import nomad.basis.trajectory as trajectory
import nomad.integrals.centroid as centroid

pes_cache  = dict()

def update_pes(master, update_integrals=True):
    """Updates the potential energy surface."""
    global pes_cache
    success = True

    # this conditional checks to see if we actually need centroids,
    # even if propagator requests them
    if update_integrals:
        glbl.master_int.update(master)

    if glbl.mpi['parallel']:
        # update electronic structure
        exec_list = []
        n_total = 0 # this ensures traj.0 is on proc 0, etc.
        for i in range(master.n_traj()):
            if master.traj[i].active and not cached(master.traj[i].label,
                                                    master.traj[i].x()):
                n_total += 1
                if n_total % glbl.mpi['nproc'] == glbl.mpi['rank']:
                    exec_list.append(master.traj[i])

        if update_integrals and glbl.master_int.require_centroids:
            # now update electronic structure in a controled way to allow for
            # parallelization
            for i in range(master.n_traj()):
                for j in range(i):
                    if glbl.master_int.centroid_required[i][j] and not \
                                             cached(integrals.centroid[i][j].label,
                                                    integrals.centroid[i][j].x()):
                        n_total += 1
                        if n_total % glbl.mpi['nproc'] == glbl.mpi['rank']:
                            exec_list.append(master.cent[i][j])

        local_results = []
        for i in range(len(exec_list)):
            if type(exec_list[i]) is trajectory.Trajectory:
                pes_calc = glbl.interface.evaluate_trajectory(exec_list[i], master.time)
            elif type(exec_list[i]) is centroid.Centroid:
                pes_calc = glbl.interface.evaluate_centroid(exec_list[i], master.time)
            else:
                raise TypeError('type='+str(type(exec_list[i]))+
                                'not recognized')
            local_results.append(pes_calc)

        global_results = glbl.mpi['comm'].allgather(local_results)

        # update the cache
        for i in range(glbl.mpi['nproc']):
            for j in range(len(global_results[i])):
                pes_cache[global_results[i][j].tag] = global_results[i][j]

        # update the bundle:
        # live trajectories
        for i in range(master.n_traj()):
            if master.traj[i].alive:
                master.traj[i].update_pes_info(pes_cache[master.traj[i].label])

        # and centroids
        if update_integrals and glbl.master_int.require_centroids:
            for i in range(master.n_traj()):
                for j in range(i):
                    c_label = glbl.master_int.centroid[i][j].label
                    if c_label in pes_cache:
                        glbl.master_int.centroid[i][j].update_pes_info(c_label)
                        glbl.master_int.centroid[j][i] = glbl.master_int.centroid[i][j]

    # if parallel overhead not worth the time and effort (eg. pes known in closed form),
    # simply run over trajectories in serial (in theory, this too could be cythonized,
    # but unlikely to ever be bottleneck)
    else:
        # iterate over trajectories..
        for i in range(master.n_traj()):
            if master.traj[i].active:
                pes_traji = glbl.interface.evaluate_trajectory(master.traj[i], master.time)
                master.traj[i].update_pes_info(pes_traji)

        # ...and centroids if need be
        if update_integrals and glbl.master_int.require_centroids:

            for i in range(master.n_traj()):
                for j in range(i):
                # if centroid not initialized, skip it
                    if glbl.master_int.centroid_required[i][j]:
                        glbl.master_int.centroid[i][j].update_pes_info(
                                          glbl.interface.evaluate_centroid(
                                          glbl.master_int.centroid[i][j], master.time))
                        glbl.master_int.centroid[j][i] = glbl.master_int.centroid[i][j]

    return success

#
#
#
def update_pes_traj(traj):
    """Updates a single trajectory

    Used during spawning.
    """
    results = None

    if glbl.mpi['rank'] == 0:
        results = glbl.interface.evaluate_trajectory(traj)

    if glbl.mpi['parallel']:
        results = glbl.mpi['comm'].bcast(results, root=0)
        glbl.mpi['comm'].barrier()

    traj.update_pes_info(results)

#
#
#
def cached(label, geom):
    """Returns True if the surface in the cache corresponds to the current
    trajectory (don't recompute the surface)."""
    global pes_cache

    if label not in pes_cache:
        return False

    dg = np.linalg.norm(geom - pes_cache[label].geom)
    if dg <= constants.fpzero:
        return True

    return False
