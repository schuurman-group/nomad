"""
Routines for handling the potential energy surface.

All calls to update the pes are localized here.  This facilitates parallel
execution of potential evaluations which is essential for ab initio PES.
"""
from functools import partial
import numpy as np
import src.fmsio.glbl as glbl

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

    if glbl.sc:
        # get the global variables from the pes interface to distribute
        # to workers
        gvars = pes.get_global_vars()
        # update electronic structure
        run_list = []
        for i in range(master.n_traj()):
            if not master.traj[i].active or cached(i, master.traj[i].x()):
                continue
            run_list.append([i, master.traj[i].x(),
                             master.traj[i].state])
        if master.ints.require_centroids:
            # update the geometries
            master.update_centroids()
            # now update electronic structure in a controled way to allow for
            # parallelization
            for i in range(master.n_cent()):
                if not master.cent[i] or cached(-i, master.cent[i].x()):
                    continue
                run_list.append([-i, master.cent[i].x(),
                                 master.cent[i].state, master.cent[i].c_state])
        jobs = glbl.sc.parallelize(run_list)
        rdd = jobs.map(partial(pes.evaluate_worker, global_var=gvars))
        res = rdd.collect()

        # update the cache
        for i, run in enumerate(run_list):
            pes_cache[run[0]] = res[i]

        # update the bundle:
        # live trajectories
        for i in range(master.n_traj()):
            if not master.traj[i].alive:
                continue
            master.traj[i].update_pes(pes_cache[i])

        # and centroids
        for i in range(master.n_cent()):
            if not master.cent[i]:
                continue
            master.cent[i].update_pes(pes_cache[-i])

    # if parallel overhead not worth the time and effort (eg. pes known in closed form),
    # simply run over trajectories in serial (in theory, this too could be cythonized,
    # but unlikely to ever be bottleneck)
    else:
        # iterate over trajectories..
        for i in range(master.n_traj()):
            if not master.traj[i].active:
                continue
            results = pes.evaluate_trajectory(i, master.traj[i].x(),
                                              master.traj[i].state)
            master.traj[i].update_pes(results)

        # ...and centroids if need be
        if master.ints.require_centroids:
            # update the geometries
            master.update_centroids()
            for i in range(master.n_cent()):
                # if centroid not initialized, skip it
                if not master.cent[i]:
                    continue
                results = pes.evaluate_centroid(i, master.cent[i].x(),
                                                master.cent[i].state,
                                                master.cent[i].c_state)
                master.cent[i].update_pes(results)

    return success


def update_pes_traj(traj):
    """Updates a single trajectory

    Used during spawning.
    """
    global pes

    results = pes.evaluate_trajectory(traj.tid, traj.x(), traj.state)
    traj.update_pes(results)


def cached(tid, geom):
    """Returns True if the surface in the cache corresponds to the current
    trajectory (don't recompute the surface)."""
    global pes_cache

    if tid not in pes_cache:
        return False

    g  = np.array([pes_cache[tid][0][i]
                   for i in range(len(pes_cache[tid][0]))])
    dg = np.linalg.norm(geom - g)
    if dg <= glbl.fpzero:
        return True

    return False
