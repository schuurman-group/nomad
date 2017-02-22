"""
Routines for handling the potential energy surface.

All calls to update the pes are localized here.  This facilitates parallel
execution of potential evaluations which is essential for ab initio PES.
"""
from functools import partial
import numpy as np
import src.fmsio.glbl as glbl
import src.basis.trajectory as trajectory
import src.basis.centroid as centroid

pes        = None
pes_cache  = dict()

def init_surface(pes_interface):
    """Initializes the potential energy surface."""
    global pes
    # create interface to appropriate surface
    try:
        pes = __import__('src.interfaces.' + pes_interface, fromlist=['NA'])
    except ImportError:
        print('INTERFACE FAIL: ' + pes_interface)

def update_pes(master):
    """Updates the potential energy surface."""
    global pes, pes_cache
    success = True

    if glbl.mpi_parallel:
        # update electronic structure
        exec_list = []
        n_total = 0 # this ensures traj.0 is on proc 0, etc.
        for i in range(master.n_traj()):
            if not master.traj[i].active or cached(master.traj[i].label, 
                                                   master.traj[i].x()):
                continue
            n_total += 1
            if n_total % glbl.mpi_nproc == glbl.mpi_rank:
                exec_list.append(master.traj[i]) 

        if master.integrals.require_centroids:
            # update the geometries
            master.update_centroids()
            # now update electronic structure in a controled way to allow for
            # parallelization
            for i in range(master.n_traj()):
                for j in range(i):
                    if not master.centroid_required(master.traj[i],master.traj[j]) or \
                                   cached(master.cent[i][j].label,master.cent[i][j].x()):
                        continue
                    n_total += 1
                    if n_total % glbl.mpi_nproc == glbl.mpi_rank:
                        exec_list.append(master.cent[i][j])

        local_results = []
        for i in range(len(exec_list)):
            if type(exec_list[i]) is trajectory.Trajectory:
                pes_calc = pes.evaluate_trajectory(exec_list[i])
            elif type(exec_list[i]) is centroid.Centroid:
                pes_calc = pes.evaluate_centroid(exec_list[i])
            else:
                sys.exit("ERROR in surface.update_pes: type="+
                         str(type(exec_list[i]))+" not recognized")
            local_results.append(pes_calc)

        global_results = glbl.mpi_comm.allgather(local_results)

        # update the cache
        for i in range(glbl.mpi_nproc):
            for j in range(len(global_results[i])):
                pes_cache[global_results[i][j].tag] = global_results[i][j]

        # update the bundle:
        # live trajectories
        for i in range(master.n_traj()):
            if not master.traj[i].alive:
                continue
            master.traj[i].update_pes_info(pes_cache[master.traj[i].label])

        # and centroids
        if master.integrals.require_centroids:
            for i in range(master.n_traj()):
                for j in range(i):
                    if master.cent[i][j].label not in pes_cache:
                        continue
                    master.cent[i][j].update_pes_info(pes_cache[master.cent[i][j].label])
                    master.cent[j][i] = master.cent[i][j]

    # if parallel overhead not worth the time and effort (eg. pes known in closed form),
    # simply run over trajectories in serial (in theory, this too could be cythonized,
    # but unlikely to ever be bottleneck)
    else:
        # iterate over trajectories..
        for i in range(master.n_traj()):
            if not master.traj[i].active:
                continue
            master.traj[i].update_pes_info(pes.evaluate_trajectory(master.traj[i]))

        # ...and centroids if need be
        if master.integrals.require_centroids:
            # update the geometries
            master.update_centroids()
            for i in range(master.n_traj()):
                for j in range(i):
                # if centroid not initialized, skip it
                    if master.cent[i][j] is None:
                        continue
                    master.cent[i][j].update_pes_info(
                                      pes.evaluate_centroid(master.cent[i][j]))
                    master.cent[j][i] = master.cent[i][j]

    return success


def update_pes_traj(traj):
    """Updates a single trajectory

    Used during spawning.
    """
    global pes

    results = None

    if glbl.mpi_rank == 0:
        results = pes.evaluate_trajectory(traj)

    if glbl.mpi_parallel:
        results = glbl.mpi_comm.bcast(results, root=0)
        glbl.mpi_comm.barrier()
    
    traj.update_pes_info(results)

def cached(label, geom):
    """Returns True if the surface in the cache corresponds to the current
    trajectory (don't recompute the surface)."""
    global pes_cache

    if label not in pes_cache:
        return False

    dg = np.linalg.norm(geom - pes_cache[label].geom)
    if dg <= glbl.fpzero:
        return True

    return False
