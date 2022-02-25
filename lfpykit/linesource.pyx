#cython: language_level=3
import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange

# DTYPE = np.float64
ctypedef np.float64_t DTYPE_t
ctypedef Py_ssize_t   LTYPE_t

cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] calc_lfp_linesource(
        np.ndarray[DTYPE_t, ndim=2, negative_indices=False] cell_x,
        np.ndarray[DTYPE_t, ndim=2, negative_indices=False] cell_y,
        np.ndarray[DTYPE_t, ndim=2, negative_indices=False] cell_z,
        DTYPE_t x,
        DTYPE_t y,
        DTYPE_t z,
        DTYPE_t sigma,
        np.ndarray[DTYPE_t, ndim=1, negative_indices=False] r_limit):
    """Calculate electric field potential using the line-source method, all
    segments treated as line sources.

    Parameters
    ----------
    cell_x: ndarray
        shape ``(totnsegs, 2)`` array with ``CellGeometry.x`` datas
    cell_y: ndarray
        shape ``(totnsegs, 2)`` array with ``CellGeometry.y`` datas
    cell_z: ndarray
        shape ``(totnsegs, 2)`` array with ``CellGeometry.z`` datas
    x: float
        extracellular position, x-axis
    y: float
        extracellular position, y-axis
    z: float
        extracellular position, z-axis
    sigma: float
        extracellular conductivity
    r_limit: np.ndarray
        minimum distance to source current for each compartment
    """

    # cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] xstart, xend, ystart, yend, zstart, zend

    # some variables for h, r2, r_root calculations
    # xstart = cell_x[:, 0]
    # xend = cell_x[:, 1]
    # ystart = cell_y[:, 0]
    # yend = cell_y[:, 1]
    # zstart = cell_z[:, 0]
    # zend = cell_z[:, 1]

    return _calc_lfp_linesource(
        cell_x[:, 0],
        cell_x[:, 1],
        cell_y[:, 0],
        cell_y[:, 1],
        cell_z[:, 0],
        cell_z[:, 1],
        x,
        y,
        z,
        sigma,
        r_limit)


cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] _calc_lfp_linesource(
                         np.ndarray[DTYPE_t, ndim=1, negative_indices=False] xstart,
                         np.ndarray[DTYPE_t, ndim=1, negative_indices=False] xend,
                         np.ndarray[DTYPE_t, ndim=1, negative_indices=False] ystart,
                         np.ndarray[DTYPE_t, ndim=1, negative_indices=False] yend,
                         np.ndarray[DTYPE_t, ndim=1, negative_indices=False] zstart,
                         np.ndarray[DTYPE_t, ndim=1, negative_indices=False] zend,
                         DTYPE_t x,
                         DTYPE_t y,
                         DTYPE_t z,
                         DTYPE_t sigma,
                         np.ndarray[DTYPE_t, ndim=1, negative_indices=False] r_limit):

    cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] deltaS, h, r2, l_, mapping
    cdef np.ndarray[LTYPE_t, ndim=1, negative_indices=False] too_close_idxs, i, ii, iii
    cdef np.ndarray[np.uint8_t, ndim = 1, cast=True] hnegi, hposi, lnegi, lposi

    deltaS = _deltaS_calc(xstart, xend, ystart, yend, zstart, zend)
    h = _h_calc(xstart, xend, ystart, yend, zstart, zend, deltaS, x, y, z)
    r2 = _r2_calc(xend, yend, zend, x, y, z, h)

    too_close_idxs = np.where(r2 < r_limit * r_limit)[0]
    r2[too_close_idxs] = r_limit[too_close_idxs]**2
    l_ = h + deltaS

    hnegi = h < 0
    hposi = h >= 0
    lnegi = l_ < 0
    lposi = l_ >= 0

    mapping = np.zeros(xstart.size)

    # case i, h < 0, l < 0, see Eq. C.13 in Gary Holt's thesis, 1998.
    [i] = np.where(hnegi & lnegi)
    # case ii, h < 0, l >= 0
    [ii] = np.where(hnegi & lposi)
    # case iii, h >= 0, l >= 0
    [iii] = np.where(hposi & lposi)

    mapping[i] = _linesource_calc_case1(l_[i], r2[i], h[i])
    mapping[ii] = _linesource_calc_case2(l_[ii], r2[ii], h[ii])
    mapping[iii] = _linesource_calc_case3(l_[iii], r2[iii], h[iii])
    return 1 / (4 * np.pi * sigma * deltaS) * mapping


@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] _linesource_calc_case1(np.ndarray[DTYPE_t, ndim=1, negative_indices=False] l_i,
                           np.ndarray[DTYPE_t, ndim=1, negative_indices=False]r2_i,
                           np.ndarray[DTYPE_t, ndim=1, negative_indices=False]h_i):
    """Calculates linesource contribution for case i"""
    cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] bb, cc
    bb = np.sqrt(h_i * h_i + r2_i) - h_i
    cc = np.sqrt(l_i * l_i + r2_i) - l_i
    return np.log(bb / cc)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] _linesource_calc_case2(np.ndarray[DTYPE_t, ndim=1, negative_indices=False] l_ii,
                           np.ndarray[DTYPE_t, ndim=1, negative_indices=False] r2_ii,
                           np.ndarray[DTYPE_t, ndim=1, negative_indices=False] h_ii):
    """Calculates linesource contribution for case ii"""
    cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] bb, cc
    bb = np.sqrt(h_ii * h_ii + r2_ii) - h_ii
    cc = (l_ii + np.sqrt(l_ii * l_ii + r2_ii)) / r2_ii
    return np.log(bb * cc)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] _linesource_calc_case3(np.ndarray[DTYPE_t, ndim=1, negative_indices=False] l_iii,
                           np.ndarray[DTYPE_t, ndim=1, negative_indices=False] r2_iii,
                           np.ndarray[DTYPE_t, ndim=1, negative_indices=False] h_iii):
    """Calculates linesource contribution for case iii"""
    cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] bb, cc
    bb = np.sqrt(l_iii * l_iii + r2_iii) + l_iii
    cc = np.sqrt(h_iii * h_iii + r2_iii) + h_iii
    return np.log(bb / cc)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] _deltaS_calc(np.ndarray[DTYPE_t, ndim=1, negative_indices=False] xstart,
                 np.ndarray[DTYPE_t, ndim=1, negative_indices=False] xend,
                 np.ndarray[DTYPE_t, ndim=1, negative_indices=False] ystart,
                 np.ndarray[DTYPE_t, ndim=1, negative_indices=False] yend,
                 np.ndarray[DTYPE_t, ndim=1, negative_indices=False] zstart,
                 np.ndarray[DTYPE_t, ndim=1, negative_indices=False] zend):
    """Returns length of each segment"""
    cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] deltaS
    deltaS = np.sqrt((xstart - xend)**2 +
                     (ystart - yend)**2 +
                     (zstart - zend)**2)
    return deltaS


@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] _h_calc(np.ndarray[DTYPE_t, ndim=1, negative_indices=False] xstart,
            np.ndarray[DTYPE_t, ndim=1, negative_indices=False] xend,
            np.ndarray[DTYPE_t, ndim=1, negative_indices=False] ystart,
            np.ndarray[DTYPE_t, ndim=1, negative_indices=False] yend,
            np.ndarray[DTYPE_t, ndim=1, negative_indices=False] zstart,
            np.ndarray[DTYPE_t, ndim=1, negative_indices=False] zend,
            np.ndarray[DTYPE_t, ndim=1, negative_indices=False] deltaS,
            DTYPE_t x,
            DTYPE_t y,
            DTYPE_t z):
    """Subroutine used by calc_lfp_*()"""
    cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] ccX, ccY, ccZ, cc
    ccX = (x - xend) * (xend - xstart)
    ccY = (y - yend) * (yend - ystart)
    ccZ = (z - zend) * (zend - zstart)
    cc = ccX + ccY + ccZ

    return cc / deltaS


@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] _r2_calc(np.ndarray[DTYPE_t, ndim=1, negative_indices=False] xend,
             np.ndarray[DTYPE_t, ndim=1, negative_indices=False] yend,
             np.ndarray[DTYPE_t, ndim=1, negative_indices=False] zend,
             DTYPE_t x,
             DTYPE_t y,
             DTYPE_t z,
             np.ndarray[DTYPE_t, ndim=1, negative_indices=False] h):
    """Subroutine used by calc_lfp_*()"""
    cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] r2
    r2 = (xend - x)**2 + (yend - y)**2 + (zend - z)**2 - h**2
    return np.abs(r2)


cpdef np.ndarray[DTYPE_t, ndim=2, negative_indices=False] _get_transform(
        np.ndarray[DTYPE_t, ndim=2, negative_indices=False] cell_x,
        np.ndarray[DTYPE_t, ndim=2, negative_indices=False] cell_y,
        np.ndarray[DTYPE_t, ndim=2, negative_indices=False] cell_z,
        np.ndarray[DTYPE_t, ndim=1, negative_indices=False] x,
        np.ndarray[DTYPE_t, ndim=1, negative_indices=False] y,
        np.ndarray[DTYPE_t, ndim=1, negative_indices=False] z,
        DTYPE_t sigma,
        np.ndarray[DTYPE_t, ndim=1, negative_indices=False] r_limit):

    cdef np.ndarray[DTYPE_t, ndim=2, negative_indices=False] M
    cdef int j, x_size
    x_size = x.size
    M = np.empty((x.size, cell_x.shape[0]))
    for j in range(x_size):
        M[j, :] = calc_lfp_linesource(
            cell_x=cell_x,
            cell_y=cell_y,
            cell_z=cell_z,
            x=x[j],
            y=y[j],
            z=z[j],
            sigma=sigma,
            r_limit=r_limit)
    return M
