"""
Linear algebra library routines.
"""
import numpy as np
import nomad.core.glbl as glbl
import nomad.math.constants as constants

def normalize(vec):
    """function that returns a normalized copy of vec"""
    norm = np.linalg.norm(v)
    if norm == constants.fpzero: 
       return vec.copy()
    return vec.copy() / norm


def pseudo_inverse(mat):
    """Modified version of the scipy pinv function.

    Altered such that the the cutoff for singular values can be set to
    a hard value. Note that by default the scipy cutoff of
    1e-15*sigma_max is taken."""
    dim1, dim2 = mat.shape

    invmat = np.zeros((dim1, dim2), dtype=complex)
    cmat=np.conjugate(mat)

    # SVD of the overlap matrix
    u, s, vt = np.linalg.svd(cmat, full_matrices=True)

    #print("\n",s,"\n")

    # Condition number
    ns = min(dim1, dim2)
    if s[ns-1] < 1e-90:
        cond = 1e+90
    else:
        cond = s[0]/s[ns-1]

    # Moore-Penrose pseudo-inverse
    if glbl.properties['sinv_thrsh'] == -1.0:
        # set cutoff to machine epsilon * sigma_max
        cutoff = np.finfo(float).eps * np.maximum.reduce(s)
    else:
        cutoff = glbl.properties['sinv_thrsh']
    for i in range(min(dim1, dim2)):
        if s[i] > cutoff:
            s[i] = 1./s[i]
        else:
            s[i] = 0.
    invmat = np.dot(np.transpose(vt), np.multiply(s[:, np.newaxis],
                                                  np.transpose(u)))

    return invmat, cond


def pseudo_inverse2(mat):
    """Modified version of the scipy pinv function.

    Altered such that the the cutoff for singular values can be set to
    a hard value. Note that by default the scipy cutoff of
    1e-15*sigma_max is taken."""
    dim1, dim2 = mat.shape

    invmat = np.zeros((dim1, dim2), dtype=complex)
    cmat=np.conjugate(mat)

    # SVD of the overlap matrix
    u, s, vt = np.linalg.svd(cmat, full_matrices=True)

    #print("\n",s,"\n")

    # Condition number
    ns = min(dim1, dim2)
    if s[ns-1] < 1e-90:
        cond = 1e+90
    else:
        cond = s[0]/s[ns-1]

    cutoff = 1e-10
    for i in range(min(dim1, dim2)):
         s[i] = 1./(s[i] + 1e-7*np.exp(-s[i] * 1e7))
         #if s[i] > cutoff:
         #    s[i] = 1./s[i]
         #else:
         #    s[i] = s + 1e-7*np.exp(-s * 1e7)

    invmat = np.dot(np.transpose(vt), np.multiply(s[:, np.newaxis],
                                                  np.transpose(u)))

    return invmat, cond
