import sys
import copy
import numpy as np
import scipy.linalg as sp_linalg
import src.utils.timings as timings
import src.integrals.centroid as centroid

class Integral:
    """Class constructor for the Bundle object."""
    def __init__(self, ansatz):
        self.type     = ansatz
        self.centroid = []
        self.centroid_required = []
        try:
            self.ints =__import__('src.integrals.'+str(self.type),fromlist=['a'])
        except ImportError:
            print('Cannot import integrals: src.integrals.'+str(self.type))

        self.hermitian         = self.ints.hermitian
        self.require_centroids = self.ints.require_centroids

    #
    #
    #
    @timings.timed
    def elec_overlap(self, bra_traj, ket_traj):
        args   = [bra_traj, ket_traj]
        kwargs = dict()
        return self.ints.elec_overlap(*args, **kwargs)

    #
    #
    #
    @timings.timed
    def nuc_overlap(self, bra_traj, ket_traj):
        args   = [bra_traj, ket_traj]
        kwargs = dict()
        return self.ints.nuc_overlap(*args, **kwargs)


    #
    #
    #
    @timings.timed
    def traj_overlap(self, bra_traj, ket_traj, nuc_ovrlp=None):
        args   = [bra_traj, ket_traj]
        kwargs = dict(nuc_ovrlp = nuc_ovrlp)
        return self.ints.traj_overlap(*args, **kwargs)

    #
    #
    #
    @timings.timed
    def s_integral(self, bra_traj, ket_traj, nuc_ovrlp=None):
        args   = [bra_traj, ket_traj]
        kwargs = dict(nuc_ovrlp = nuc_ovrlp)
        return self.ints.s_integral(*args, **kwargs)    

    #
    #
    #
    @timings.timed
    def t_integral(self, bra_traj, ket_traj, nuc_ovrlp=None):
        args   = [bra_traj, ket_traj]
        kwargs = dict(nuc_ovrlp = nuc_ovrlp)
        return self.ints.t_integral(*args, **kwargs)

    #
    #
    #
    @timings.timed
    def v_integral(self, bra_traj, ket_traj, nuc_ovrlp=None):
        args   = [bra_traj, ket_traj]
        kwargs = dict(nuc_ovrlp = nuc_ovrlp)
        if self.require_centroids:
            args.append(self.centroid[bra_traj.label][ket_traj.label])
        return self.ints.v_integral(*args, **kwargs)

    #
    #
    #
    @timings.timed
    def sdot_integral(self, bra_traj, ket_traj, nuc_ovrlp=None):
        args   = [bra_traj, ket_traj]
        kwargs = dict(nuc_ovrlp = nuc_ovrlp)
        return self.ints.sdot_integral(*args, **kwargs)

    #
    #
    #
    def wfn_overlap(self, bra_wfn, ket_wfn):
        """Documentation to come"""

        S = 0.
        for i in range(bra_wfn.nalive):
            for j in range(ket_wfn.nalive):
                ii = bra_wfn.alive[i]
                jj = ket_wfn.alive[j]
                S += (self.traj_overlap(bra_wfn.traj[ii], ket_wfn.traj[jj]) *
                                        bra_wfn.traj[ii].amplitude.conjugate() *
                                        ket_wfn.traj[jj].amplitude)
        return S

    #
    #
    #
    def wfn_project(self, bra_traj, ket_wfn):
        """Returns the overlap of the wfn with a trajectory (assumes the
        amplitude on the trial trajectory is (1.,0.)"""
        proj = 0j

        for i in range(ket_wfn.nalive + ket_wfn.ndead):
            proj += self.traj_overlap(bra_traj, ket_wfn.traj[i]) * ket_wfn.traj[i].amplitude

        return proj

    #
    #
    #
    def update(self, wfn):

        if self.ints.require_centroids:
            self.update_centroids(wfn)

        return

    #
    #
    #
    def add_centroid(self, new_cent):
        """places the centroid in a centroid array -- increases the array
           if necessary in order to accomodate data"""

        # minimum dimension required to hold this centroid
        ij           = new_cent.parents
        new_dim_cent = max(ij)
        dim_cent     = len(self.centroid)

        # if current array is too small, expand by necessary number of dimensions
        if new_dim_cent > dim_cent:
            for i in range(dim_cent):
                self.centroid[i].extend([None for j in range(new_dim_cent -
                                                                 dim_cent)])

            for i in range(new_dim_cent - dim_cent):
                self.centroid.append([None for j in range(new_dim_cent)])

        self.centroid[ij[0]][ij[1]] = new_cent
        self.centroid[ij[1]][ij[0]] = new_cent

        return



#------------------------------------------------------
#
#  Private Functions
#
    @timings.timed
    def update_centroids(self, wfn):
        """Increases the centroid 'matrix' to account for new basis functions.

        Called by add_trajectory. Make sure centroid array has sufficient
        space to hold required centroids. Note that n_traj includes alive
        AND dead trajectories -- therefore it can only increase. So, only
        need to check n_traj > dim_cent condition.
        """
        dim_cent = len(self.centroid)

        # number of centroids already correct
        if wfn.n_traj() == dim_cent:
            return

        # n_traj includes living and dead -- this condition should never be satisfied
        if wfn.n_traj() < dim_cent:
            raise ValueError('n_traj() < dim_cent in wfn. Exiting...')

        # ...else we need to add more centroids
        if self.n_traj() > dim_cent:
            for i in range(dim_cent):
                self.centroid[i].extend([None for j in range(self.n_traj() -
                                                         dim_cent)])

            for i in range(self.n_traj() - dim_cent):
                self.centroid.append([None for j in range(self.n_traj())])

        for i in range(wfn.n_traj()):
            for j in range(i):
                # now check to see if needed index has an existing trajectory
                # if not, copy trajectory from one of the parents into the
                # required slots
                if wfn.cent[i][j] is None and (wfn.traj[i].alive and wfn.traj[j].alive):
                    wfn.cent[i][j] = centroid.Centroid(traj_i=wfn.traj[i],
                                                       traj_j=wfn.traj[j])
                    self.centroid[i][j].update_x(wfn.traj[i],wfn.traj[j])
                    self.centroid[i][j].update_p(wfn.traj[i],wfn.traj[j])
                    self.centroid[j][i] = self.centroid[i][j]
                self.centroid_required[i][j] = self.isRequired(wfn.traj[i],wfn.traj[j])
                self.centroid_required[i][j] = self.isRequired(wfn.traj[j],wfn.traj[i])
 
        return

    #
    #
    #
    def isRequired(traj1, traj2):
        """Documentation to come"""
        return (traj1.alive and traj2.alive) 


