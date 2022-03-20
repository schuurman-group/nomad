"""
Routines for computing integrals.
"""
import numpy as np
import nomad.core.timings as timings
import nomad.integrals.centroid as centroid


class Integral:
    """Class constructor for the Bundle object."""
    def __init__(self, kecoef, ansatz, numerical_eval):
        self.kecoef    = kecoef
        self.ansatz    = ansatz
        self.numerical = numerical_eval
        self.centroids = []
        self.centroid_required = []

        # some logic to determine how to evaluate integrals
        # over basis functions.
        # ansatz can be: fms, mce, mca, diabatic
        # numerical can: bat, saddle_point, vibronic, dirac
        if self.numerical == 'dirac':
            self.ints      = __import__('nomad.integrals.'+str(self.ansatz)+
                                        '_'+str(self.numerical), fromlist=['a'])
            self.ints_eval = __import__('nomad.integrals.'+str(self.ansatz)+
                                        '_'+str(self.numerical),fromlist=['a'])
        else:
            self.ints_eval = __import__('nomad.integrals.'+str(self.ansatz)+
                                        '_'+str(self.numerical),fromlist=['a'])

            # this is a temporary hack
            if str(self.numerical) != 'lvc_exact':
                self.ints = __import__('nomad.integrals.'+str(self.ansatz),
                                        fromlist=['a'])
            else:
                self.ints = self.ints_eval

        self.hermitian            = self.ints.hermitian
        self.require_centroids    = self.ints_eval.require_centroids
        self.overlap_requires_pes = False

    @timings.timed
    def elec_overlap(self, bra_traj, ket_traj):
        """Calculates the electronic overlap."""
        return self.ints.elec_overlap(bra_traj, ket_traj)

    @timings.timed
    def nuc_overlap(self, bra_traj, ket_traj):
        """Calculates the nuclear overlap."""
        return self.ints.nuc_overlap(bra_traj, ket_traj)

    @timings.timed
    def traj_overlap(self, bra_traj, ket_traj):
        """Calculates the trajectory overlap."""
        return self.ints.traj_overlap(bra_traj, ket_traj)

    @timings.timed
    def s_integral(self, bra_traj, ket_traj, nuc_ovrlp=None, elec_ovrlp=None):
        """Calculates the overlap integral between two trajectories."""
        if nuc_ovrlp is None:
            nuc_ovrlp = self.ints.nuc_overlap(bra_traj, ket_traj)

        if elec_ovrlp is None:
            elec_ovrlp = self.ints.elec_overlap(bra_traj, ket_traj)

        return self.ints.s_integral(bra_traj, ket_traj, nuc_ovrlp, elec_ovrlp)

    @timings.timed
    def t_integral(self, bra_traj, ket_traj, nuc_ovrlp=None, elec_ovrlp=None):
        """Calculates the kinetic energy integral between two trajectories."""
        if nuc_ovrlp is None:
            nuc_ovrlp = self.ints.nuc_overlap(bra_traj, ket_traj)

        if elec_ovrlp is None:
            elec_ovrlp = self.ints.elec_overlap(bra_traj, ket_traj)

        return self.ints.t_integral(bra_traj, ket_traj, nuc_ovrlp, elec_ovrlp)

    @timings.timed
    def v_integral(self, bra_traj, ket_traj, nuc_ovrlp=None, elec_ovrlp=None):
        """Calculates the potential energy integral between two
        trajectories."""
        if nuc_ovrlp is None:
            nuc_ovrlp = self.ints.nuc_overlap(bra_traj, ket_traj)

        if elec_ovrlp is None:
            elec_ovrlp = self.ints.elec_overlap(bra_traj, ket_traj)

        if self.require_centroids:
            return self.ints_eval.v_integral(bra_traj, ket_traj,
                                        self.centroids[bra_traj.label][ket_traj.label],
                                             nuc_ovrlp, elec_ovrlp)
        else:
            return self.ints_eval.v_integral(bra_traj, ket_traj,
                                             nuc_ovrlp, elec_ovrlp)

    @timings.timed
    def sdot_integral(self, bra_traj, ket_traj, nuc_ovrlp=None, elec_ovrlp=None):
        """Calculates the time derivative of the nuclear overlap."""
        if nuc_ovrlp is None:
            nuc_ovrlp = self.ints.nuc_overlap(bra_traj, ket_traj)

        if elec_ovrlp is None:
            elec_ovrlp = self.ints.elec_overlap(bra_traj, ket_traj)

        return self.ints.sdot_integral(bra_traj, ket_traj, nuc_ovrlp, elec_ovrlp)


    @timings.timed
    def popwt_integral(self, bra_traj, ket_traj, nuc_ovrlp=None):
        """Calculates the population contribution to each state for pair of trajectories."""
        if nuc_ovrlp is None:
            nuc_ovrlp = self.ints.nuc_overlap(bra_traj, ket_traj)

        return self.ints.popwt(bra_traj, ket_traj, nuc_ovrlp)


    def wfn_overlap(self, bra_wfn, ket_wfn):
        """Calculates the overall wavefunction overlap."""
        S = 0.
        for i in range(bra_wfn.nalive):
            for j in range(ket_wfn.nalive):
                ii = bra_wfn.alive[i]
                jj = ket_wfn.alive[j]
                S += (self.traj_overlap(bra_wfn.traj[ii], ket_wfn.traj[jj]) *
                                        bra_wfn.traj[ii].amplitude.conjugate() *
                                        ket_wfn.traj[jj].amplitude)
        return S

    def wfn_project(self, bra_traj, ket_wfn):
        """Returns the overlap of the wfn with a trajectory.

        Assumes the amplitude on the trial trajectory is (1.,0.)
        """
        proj = 0j

        for i in range(ket_wfn.nalive + ket_wfn.ndead):
            proj += self.traj_overlap(bra_traj, ket_wfn.traj[i]) * ket_wfn.traj[i].amplitude

        return proj

    def update(self, wfn):
        """Updates the wavefunction information if required."""
        if self.require_centroids:
            self.update_centroids(wfn)

    def add_centroid(self, new_cent):
        """Places the centroid in a centroid array.

        Increases the array size if necessary in order to accomodate data.
        """
        # minimum dimension required to hold this centroid
        ij           = new_cent.parents
        new_dim_cent = max(ij)+1 # index of last trajectory is traj_id, dimension == traj_id+1
        dim_cent     = len(self.centroids)

        # if current array is too small, expand by necessary number of dimensions
        if new_dim_cent > dim_cent:
            for i in range(dim_cent):
                self.centroids[i].extend([None for j in range(new_dim_cent -
                                                                 dim_cent)])
            self.centroids.extend([[None for j in range(new_dim_cent)]
                                         for k in range(new_dim_cent-dim_cent)])

        self.centroids[ij[0]][ij[1]] = new_cent
        self.centroids[ij[1]][ij[0]] = new_cent

    #------------------------------------------------------
    #
    #  Private Methods
    #
    @timings.timed
    def update_centroids(self, wfn):
        """Increases the centroid 'matrix' to account for new basis functions.

        Called by add_trajectory. Make sure centroid array has sufficient
        space to hold required centroids. Note that n_traj includes alive
        AND dead trajectories -- therefore it can only increase. So, only
        need to check n_traj > dim_cent condition.
        """

        n_traj   = wfn.n_traj()
        dim_cent = len(self.centroids)
        dim_req  = len(self.centroid_required)

        # n_traj includes living and dead -- this condition should never be satisfied
        if wfn.n_traj() < dim_cent:
            raise ValueError('n_traj() < dim_cent in wfn. Exiting...')

        if dim_cent < n_traj:
            for i in range(dim_cent):
                self.centroids[i].extend([None for j in range(n_traj - dim_cent)])
            self.centroids.extend([[None for j in range(n_traj)] for k in range(n_traj - dim_cent)])
        if dim_req < n_traj:
            for i in range(dim_req):
                self.centroid_required[i].extend([None for j in range(n_traj - dim_req)])
            self.centroid_required.extend([[None for j in range(n_traj)] for k in range(n_traj - dim_req)])

        for i in range(n_traj):
            for j in range(i):
                # now check to see if needed index has an existing trajectory
                # if not, copy trajectory from one of the parents into the
                # required slots
                if self.centroids[i][j] is None and (wfn.traj[i].alive and wfn.traj[j].alive):
                    self.centroids[i][j] = centroid.Centroid(traj_i=wfn.traj[i],
                                                            traj_j=wfn.traj[j])
                self.centroids[i][j].update_x(wfn.traj[i],wfn.traj[j])
                self.centroids[i][j].update_p(wfn.traj[i],wfn.traj[j])
                self.centroids[j][i] = self.centroids[i][j]
                self.centroid_required[i][j] = self.is_required(wfn.traj[i],wfn.traj[j])
                self.centroid_required[j][i] = self.centroid_required[i][j]

    def is_required(self, traj1, traj2):
        """Documentation to come"""
        return traj1.alive and traj2.alive
