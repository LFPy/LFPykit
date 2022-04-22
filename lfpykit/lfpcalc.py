#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Copyright (C) 2012 Computational Neuroscience Group, NMBU.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

"""

import numpy as np


def return_dist_from_segments(xstart, ystart, zstart, xend, yend, zend, p):
    """
    Returns distance and closest point on line-segments from point p

    Parameters
    ----------
    xstart: ndarray
        start points of segments along x-axis
    ystart: ndarray
        start points of segments along y-axis
    zstart: ndarray
        start point of segments along z-axis
    xend: ndarray
        end points of segments along x-axis
    yend: ndarray
        end points of segments along y-axis
    zend: ndarray
        end points of segments along z-axis
    p: ndarray
        position of contact

    Returns
    -------
    dist: ndarray
        distance to the closest point in line
    closest_point: ndarray
        the closest point on the line
    """
    px = xend - xstart
    py = yend - ystart
    pz = zend - zstart

    delta = px * px + py * py + pz * pz
    u = ((p[0] - xstart) * px + (p[1] - ystart) * py +
         (p[2] - zstart) * pz) / delta
    u[u > 1] = 1.0
    u[u < 0] = 0.0

    closest_point = np.array([xstart + u * px,
                              ystart + u * py,
                              zstart + u * pz])
    dist = np.sqrt(np.sum((closest_point.T - p)**2, axis=1))
    return dist, closest_point


def calc_lfp_linesource_anisotropic(cell_x, cell_y, cell_z,
                                    x, y, z, sigma, r_limit, atol=1e-8):
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
    sigma: array
        extracellular conductivity [sigma_x, sigma_y, sigma_z]
    r_limit: np.ndarray
        minimum distance to source current for each compartment
    atol: float
        numerical (absolute) tolerance for near-zero comparisons.
        Should be a small number, and should normally not need to be changed.
    """

    xstart = cell_x[:, 0]
    xend = cell_x[:, -1]
    ystart = cell_y[:, 0]
    yend = cell_y[:, -1]
    zstart = cell_z[:, 0]
    zend = cell_z[:, -1]
    px = xend - xstart
    py = yend - ystart
    pz = zend - zstart
    pos = np.array([x, y, z])
    rs, closest_points = return_dist_from_segments(xstart, ystart, zstart,
                                                   xend, yend, zend, pos)

    # If measurement point is too close we move it further away
    dr_ = np.array([pos[0] - xstart,  pos[1] - ystart, pos[2] - zstart])
    r0_ = np.array([xstart, ystart, zstart])
    displace_vecs = np.zeros((3, len(xstart)))

    idxs_1 = (rs < r_limit) & (rs >= atol)
    idxs_2 = (rs < r_limit) & (rs < atol) & (np.abs(px) <= atol)
    idxs_3 = (rs < r_limit) & (rs < atol) & (np.abs(px) > atol)

    if np.any(idxs_1):
        # move point in radial direction from the line segment
        displace_vecs[:, idxs_1] = (pos[:, None] - closest_points[:, idxs_1])
    if np.any(idxs_2):
        # point is directly on line-segment, and we move it in a perpendicular
        # direction. If px is zero, perpendicular direction is found from y,z
        displace_vecs[:, idxs_2] = np.array([np.zeros(np.sum(idxs_2)),
                                             pz[idxs_2],
                                             -py[idxs_2]])
    if np.any(idxs_3):
        # point is directly on line-segment, and we move it in a perpendicular
        # direction. Perpendicular direction is found from x,y
        displace_vecs[:, idxs_3] = np.array([-py[idxs_3],
                                             px[idxs_3],
                                             np.zeros(np.sum(idxs_3))])

    idxs = idxs_1 + idxs_2 + idxs_3
    displace_vecs[:, idxs] /= np.linalg.norm(displace_vecs[:, idxs], axis=0)
    dr_[:, idxs] = (closest_points[:, idxs] +
                    displace_vecs[:, idxs] * r_limit[idxs] - r0_[:, idxs])

    a = (sigma[1] * sigma[2] * px**2
         + sigma[0] * sigma[2] * py**2
         + sigma[0] * sigma[1] * pz**2)
    b = -2 * (sigma[1] * sigma[2] * dr_[0, :] * px +
              sigma[0] * sigma[2] * dr_[1, :] * py +
              sigma[0] * sigma[1] * dr_[2, :] * pz)
    c = (sigma[1] * sigma[2] * dr_[0, :]**2 +
         sigma[0] * sigma[2] * dr_[1, :]**2 +
         sigma[0] * sigma[1] * dr_[2, :]**2)

    i = np.abs(b) <= atol
    iia = (np.abs(4 * a * c - b * b) < atol) & (np.abs(a - c) < atol)
    iib = (np.abs(4 * a * c - b * b) < atol) & (np.abs(a - c) >= atol)
    iii = (4 * a * c - b * b < -atol) & (np.abs(b) > atol)
    iiii = (4 * a * c - b * b > atol) & (np.abs(b) > atol)

    # Consistency check that all indexes has been accounted for:
    if not np.all(i + iia + iib + iii + iiii):
        raise RuntimeError("Not all indexes were accounted for!")

    mapping = np.zeros(xstart.size)
    mapping[i] = _anisotropic_line_source_case_i(a[i], c[i])
    mapping[iia] = _anisotropic_line_source_case_iia(a[iia], c[iia])
    mapping[iib] = _anisotropic_line_source_case_iib(a[iib], b[iib], c[iib])
    mapping[iii] = _anisotropic_line_source_case_iii(a[iii], b[iii], c[iii])
    mapping[iiii] = _anisotropic_line_source_case_iiii(a[iiii], b[iiii],
                                                       c[iiii])

    if np.isnan(mapping).any():
        raise RuntimeError("NaN")

    return 1 / (4 * np.pi) * mapping / np.sqrt(a)


def calc_lfp_root_as_point_anisotropic(cell_x, cell_y, cell_z,
                                       x, y, z, sigma, r_limit, atol=1e-8):
    """Calculate electric field potential, root is treated as point source, all
    segments except root are treated as line sources.

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
    sigma: array
        extracellular conductivity [sigma_x, sigma_y, sigma_z]
    r_limit: np.ndarray
        minimum distance to source current for each compartment
    atol: float
        numerical (absolute) tolerance for near-zero comparisons.
        Should be a small number, and should normally not need to be changed.
    """

    xstart = cell_x[:, 0]
    xend = cell_x[:, -1]
    ystart = cell_y[:, 0]
    yend = cell_y[:, -1]
    zstart = cell_z[:, 0]
    zend = cell_z[:, -1]
    px = xend - xstart
    py = yend - ystart
    pz = zend - zstart
    # First we do line sources
    pos = np.array([x, y, z])
    rs, closest_points = return_dist_from_segments(xstart, ystart, zstart,
                                                   xend, yend, zend, pos)

    # If measurement point is too close we move it further away
    dr_ = np.array([pos[0] - xstart,  pos[1] - ystart, pos[2] - zstart])
    r0_ = np.array([xstart, ystart, zstart])
    displace_vecs = np.zeros((3, len(xstart)))

    idxs_1 = (rs < r_limit) & (rs >= atol)
    idxs_2 = (rs < r_limit) & (rs < atol) & (np.abs(px) <= atol)
    idxs_3 = (rs < r_limit) & (rs < atol) & (np.abs(px) > atol)

    if np.any(idxs_1):
        # move point in radial direction from the line segment
        displace_vecs[:, idxs_1] = (pos[:, None] - closest_points[:, idxs_1])
    if np.any(idxs_2):
        # point is directly on line-segment, and we move it in a perpendicular
        # direction. If px is zero, perpendicular direction is found from y,z
        displace_vecs[:, idxs_2] = np.array([np.zeros(np.sum(idxs_2)),
                                             pz[idxs_2],
                                             -py[idxs_2]])
    if np.any(idxs_3):
        # point is directly on line-segment, and we move it in a perpendicular
        # direction. Perpendicular direction is found from x,y
        displace_vecs[:, idxs_3] = np.array([-py[idxs_3],
                                             px[idxs_3],
                                             np.zeros(np.sum(idxs_3))])

    idxs = idxs_1 + idxs_2 + idxs_3
    displace_vecs[:, idxs] /= np.linalg.norm(displace_vecs[:, idxs], axis=0)
    dr_[:, idxs] = (closest_points[:, idxs] +
                    displace_vecs[:, idxs] * r_limit[idxs] - r0_[:, idxs])

    a = (sigma[1] * sigma[2] * px**2
         + sigma[0] * sigma[2] * py**2
         + sigma[0] * sigma[1] * pz**2)
    b = -2 * (sigma[1] * sigma[2] * dr_[0, :] * px +
              sigma[0] * sigma[2] * dr_[1, :] * py +
              sigma[0] * sigma[1] * dr_[2, :] * pz)
    c = (sigma[1] * sigma[2] * dr_[0, :]**2 +
         sigma[0] * sigma[2] * dr_[1, :]**2 +
         sigma[0] * sigma[1] * dr_[2, :]**2)

    i = np.abs(b) <= atol
    iia = (np.abs(4 * a * c - b * b) < atol) & (np.abs(a - c) < atol)
    iib = (np.abs(4 * a * c - b * b) < atol) & (np.abs(a - c) >= atol)
    iii = (4 * a * c - b * b < -atol) & (np.abs(b) > atol)
    iiii = (4 * a * c - b * b > atol) & (np.abs(b) > atol)

    # Consistency check that all indexes has been accounted for:
    if not np.all(i + iia + iib + iii + iiii):
        raise RuntimeError("Not all indexes were accounted for!")

    mapping = np.zeros(xstart.size)
    mapping[i] = _anisotropic_line_source_case_i(a[i], c[i])
    mapping[iia] = _anisotropic_line_source_case_iia(a[iia], c[iia])
    mapping[iib] = _anisotropic_line_source_case_iib(a[iib], b[iib], c[iib])
    mapping[iii] = _anisotropic_line_source_case_iii(a[iii], b[iii], c[iii])
    mapping[iiii] = _anisotropic_line_source_case_iiii(a[iiii], b[iiii],
                                                       c[iiii])

    if np.isnan(mapping).any():
        raise RuntimeError("NaN")
    mapping /= np.sqrt(a)

    # Now we do root source:
    # Get compartment indices for root segment (to be treated as point
    # sources)
    rootinds = np.array([0])

    dx2_root = (cell_x[rootinds, :].mean(axis=-1) - pos[0])**2
    dy2_root = (cell_y[rootinds, :].mean(axis=-1) - pos[1])**2
    dz2_root = (cell_z[rootinds, :].mean(axis=-1) - pos[2])**2

    r2_root = dx2_root + dy2_root + dz2_root

    # Go through and correct all (if any) root idxs that are too close
    # If measurement point is directly on root, we need to assign a direction:
    dx2_root[r2_root < atol] = r_limit[rootinds]**2
    r2_root[r2_root < atol] = r_limit[rootinds]**2

    close_idx = r2_root < r_limit[rootinds]**2
    r2_scale_factor = r_limit[rootinds[close_idx]]**2 / r2_root[close_idx]
    dx2_root[close_idx] *= r2_scale_factor
    dy2_root[close_idx] *= r2_scale_factor
    dz2_root[close_idx] *= r2_scale_factor

    mapping[rootinds] = 1 / np.sqrt(sigma[1] * sigma[2] * dx2_root
                                    + sigma[0] * sigma[2] * dy2_root
                                    + sigma[0] * sigma[1] * dz2_root)

    return 1. / (4 * np.pi) * mapping


def _anisotropic_line_source_case_i(a, c):
    return np.log(np.sqrt(a / c) + np.sqrt(a / c + 1))


def _anisotropic_line_source_case_iia(a, c):
    return np.log(np.abs(1 + np.sqrt(a / c)))


def _anisotropic_line_source_case_iib(a, b, c):
    return np.abs(np.log(np.abs(np.sign(b) * np.sqrt(a / c) + 1)))


def _anisotropic_line_source_case_iii(a, b, c):
    return np.log(np.abs((2 * a + b + 2 * np.sqrt(a * (a + b + c)))
                         / (b + 2 * np.sqrt(a * c))))


def _anisotropic_line_source_case_iiii(a, b, c):
    return (np.arcsinh((2 * a + b) / np.sqrt(4 * a * c - b * b)) -
            np.arcsinh(b / np.sqrt(4 * a * c - b * b)))


def calc_lfp_linesource(cell_x, cell_y, cell_z,
                        x, y, z, sigma, r_limit):
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

    # some variables for h, r2, r_root calculations
    xstart = cell_x[:, 0]
    xend = cell_x[:, 1]
    ystart = cell_y[:, 0]
    yend = cell_y[:, 1]
    zstart = cell_z[:, 0]
    zend = cell_z[:, 1]

    return _calc_lfp_linesource(
        xstart,
        xend,
        ystart,
        yend,
        zstart,
        zend,
        x,
        y,
        z,
        sigma,
        r_limit)


def _calc_lfp_linesource(xstart,
                         xend,
                         ystart,
                         yend,
                         zstart,
                         zend,
                         x,
                         y,
                         z,
                         sigma,
                         r_limit):
    deltaS = _deltaS_calc(xstart, xend, ystart, yend, zstart, zend)
    h = _h_calc(xstart, xend, ystart, yend, zstart, zend, deltaS, x, y, z)
    r2 = _r2_calc(xend, yend, zend, x, y, z, h)

    too_close_idxs = r2 < (r_limit * r_limit)
    r2[too_close_idxs] = r_limit[too_close_idxs]**2
    l_ = h + deltaS

    hnegi = h < 0
    hposi = h >= 0
    lnegi = l_ < 0
    lposi = l_ >= 0

    mapping = np.zeros(xstart.size)

    # case i, h < 0, l < 0, see Eq. C.13 in Gary Holt's thesis, 1998.
    i = hnegi & lnegi
    # case ii, h < 0, l >= 0
    ii = hnegi & lposi
    # case iii, h >= 0, l >= 0
    iii = hposi & lposi

    mapping[i] = _linesource_calc_case1(l_[i], r2[i], h[i])
    mapping[ii] = _linesource_calc_case2(l_[ii], r2[ii], h[ii])
    mapping[iii] = _linesource_calc_case3(l_[iii], r2[iii], h[iii])
    return 1 / (4 * np.pi * sigma * deltaS) * mapping


def calc_lfp_root_as_point(cell_x, cell_y, cell_z, x, y, z, sigma, r_limit,
                           rootinds=np.array([0])):
    """Calculate electric field potential using the line-source method,
    root is treated as point/sphere source

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
        extracellular conductivity in S/m
    r_limit: np.ndarray
        minimum distance to source current for each compartment.
    rootinds: ndarray, dtype=int
        indices of root segment(s). Defaults to np.array([0])
    """
    # some variables for h, r2, r_root calculations
    xstart = cell_x[:, 0]
    xmid = cell_x[rootinds, :].mean(axis=-1)
    xend = cell_x[:, -1]
    ystart = cell_y[:, 0]
    ymid = cell_y[rootinds, :].mean(axis=-1)
    yend = cell_y[:, -1]
    zstart = cell_z[:, 0]
    zmid = cell_z[rootinds, :].mean(axis=-1)
    zend = cell_z[:, -1]

    deltaS = _deltaS_calc(xstart, xend, ystart, yend, zstart, zend)
    h = _h_calc(xstart, xend, ystart, yend, zstart, zend, deltaS, x, y, z)
    r2 = _r2_calc(xend, yend, zend, x, y, z, h)
    r_root = _r_root_calc(xmid, ymid, zmid, x, y, z)
    if np.any(r_root < r_limit[rootinds]):
        print('Adjusting r-distance to root segments')
        r_root[r_root < r_limit[rootinds]
               ] = r_limit[rootinds][r_root < r_limit[rootinds]]

    # avoid denominator approaching 0
    too_close_idxs = r2 < (r_limit * r_limit)
    r2[too_close_idxs] = r_limit[too_close_idxs]**2
    l_ = h + deltaS

    hnegi = h < 0
    hposi = h >= 0
    lnegi = l_ < 0
    lposi = l_ >= 0

    # Ensuring that root is not treated as line-source
    hnegi[rootinds] = hposi[rootinds] = lnegi[rootinds] = lposi[rootinds] = \
        False

    # Line sources
    # case i,  h < 0,  l < 0
    i = hnegi & lnegi
    # case ii,  h < 0,  l >= 0
    ii = hnegi & lposi
    # case iii,  h >= 0,  l >= 0
    iii = hposi & lposi

    # Sum all potential contributions
    mapping = np.zeros(xstart.size)
    mapping[rootinds] = 1 / r_root
    deltaS[rootinds] = 1.

    mapping[i] = _linesource_calc_case1(l_[i], r2[i], h[i])
    mapping[ii] = _linesource_calc_case2(l_[ii], r2[ii], h[ii])
    mapping[iii] = _linesource_calc_case3(l_[iii], r2[iii], h[iii])

    return 1 / (4 * np.pi * sigma * deltaS) * mapping


def _linesource_calc_case1(l_i,
                           r2_i,
                           h_i):
    """Calculates linesource contribution for case i"""
    bb = np.sqrt(h_i * h_i + r2_i) - h_i
    cc = np.sqrt(l_i * l_i + r2_i) - l_i
    return np.log(bb / cc)


def _linesource_calc_case2(l_ii,
                           r2_ii,
                           h_ii):
    """Calculates linesource contribution for case ii"""
    bb = np.sqrt(h_ii * h_ii + r2_ii) - h_ii
    cc = (l_ii + np.sqrt(l_ii * l_ii + r2_ii)) / r2_ii
    return np.log(bb * cc)


def _linesource_calc_case3(l_iii,
                           r2_iii,
                           h_iii):
    """Calculates linesource contribution for case iii"""
    bb = np.sqrt(l_iii * l_iii + r2_iii) + l_iii
    cc = np.sqrt(h_iii * h_iii + r2_iii) + h_iii
    return np.log(bb / cc)


def _deltaS_calc(xstart,
                 xend,
                 ystart,
                 yend,
                 zstart,
                 zend):
    """Returns length of each segment"""
    deltaS = np.sqrt((xstart - xend)**2 +
                     (ystart - yend)**2 +
                     (zstart - zend)**2)
    return deltaS


def _h_calc(xstart,
            xend,
            ystart,
            yend,
            zstart,
            zend,
            deltaS,
            x,
            y,
            z):
    """Subroutine used by calc_lfp_*()"""
    ccX = (x - xend) * (xend - xstart)
    ccY = (y - yend) * (yend - ystart)
    ccZ = (z - zend) * (zend - zstart)
    cc = ccX + ccY + ccZ

    return cc / deltaS


def _r2_calc(xend,
             yend,
             zend,
             x,
             y,
             z,
             h):
    """Subroutine used by calc_lfp_*()"""
    r2 = (xend - x)**2 + (yend - y)**2 + (zend - z)**2 - h**2
    return np.abs(r2)


def _r_root_calc(xmid, ymid, zmid, x, y, z):
    """calculate the distance to root midpoint"""
    r_root = np.sqrt((x - xmid)**2 + (y - ymid)**2 + (z - zmid)**2)
    return r_root


def calc_lfp_pointsource(cell_x, cell_y, cell_z,
                         x, y, z, sigma, r_limit):
    """Calculate extracellular potentials using the point-source
    equation on all compartments

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
    r2 = (cell_x.mean(axis=-1) - x)**2 + \
         (cell_y.mean(axis=-1) - y)**2 + \
         (cell_z.mean(axis=-1) - z)**2
    r2 = _check_rlimit_point(r2, r_limit)
    mapping = 1. / (4. * np.pi * sigma * np.sqrt(r2))
    return mapping


def calc_lfp_pointsource_anisotropic(cell_x, cell_y, cell_z,
                                     x, y, z, sigma, r_limit, atol=1e-8):
    """Calculate extracellular potentials using the anisotropic point-source
    equation on all compartments

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
    sigma: array
        extracellular conductivity in [x,y,z]-direction
    r_limit: np.ndarray
        minimum distance to source current for each compartment
    atol: float
        numerical (absolute) tolerance for near-zero comparisons.
        Should be a small number, and should normally not need to be changed.
    """

    dx2 = (cell_x.mean(axis=-1) - x)**2
    dy2 = (cell_y.mean(axis=-1) - y)**2
    dz2 = (cell_z.mean(axis=-1) - z)**2

    r2 = dx2 + dy2 + dz2
    zero_dist_idxs = np.abs(r2) < atol
    dx2[zero_dist_idxs] = r_limit[zero_dist_idxs]**2
    r2[zero_dist_idxs] = r_limit[zero_dist_idxs]**2

    close_idxs = r2 < r_limit**2
    # For anisotropic media, the direction we move points matter
    # Radial distance between point source and electrode is scaled to r_limit
    r2_scale_factor = r_limit[close_idxs]**2 / r2[close_idxs]
    dx2[close_idxs] *= r2_scale_factor
    dy2[close_idxs] *= r2_scale_factor
    dz2[close_idxs] *= r2_scale_factor

    sigma_r = np.sqrt(sigma[1] * sigma[2] * dx2
                      + sigma[0] * sigma[2] * dy2
                      + sigma[0] * sigma[1] * dz2)

    mapping = 1 / (4 * np.pi * sigma_r)
    return mapping


def _check_rlimit_point(r2, r_limit):
    """Correct r2 so that r2 >= r_limit**2 for all values"""
    inds = r2 < r_limit * r_limit
    r2[inds] = r_limit[inds] * r_limit[inds]
    return r2


def calc_lfp_pointsource_moi(cell_x, cell_y, cell_z,
                             x, y, z,
                             sigma_T, sigma_S, sigma_G,
                             steps, h, r_limit, **kwargs):
    """Calculate extracellular potentials using the point-source
    equation on all compartments for in vitro Microelectrode Array (MEA) slices

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
    sigma_T: float
        extracellular conductivity in tissue slice
    sigma_G: float
        Conductivity of MEA glass electrode plane.
        Should normally be zero for MEA set up.
    sigma_S: float
        Conductivity of saline bath that tissue slice is immersed in
    steps: int
        Number of steps to average over the in technically infinite sum
    h: float
        Slice thickness in um.
    r_limit: np.ndarray
        minimum distance to source current for each compartment
    """
    cell_z_mid = cell_z.mean(axis=-1)
    z_ = z

    if "z_shift" in kwargs and kwargs["z_shift"] is not None:
        # Shifting coordinate system before calculations
        z_ -= kwargs["z_shift"]
        cell_z_mid -= kwargs["z_shift"]

    dx2 = (x - cell_x.mean(axis=-1))**2
    dy2 = (y - cell_y.mean(axis=-1))**2
    dz2 = (z_ - cell_z_mid)**2

    dL2 = dx2 + dy2
    # avoid denominator approaching 0
    inds = (dL2 + dz2) < (r_limit * r_limit)
    dL2[inds] = r_limit[inds] * r_limit[inds] - dz2[inds]

    def _omega(dz):
        return 1 / np.sqrt(dL2 + dz * dz)

    WTS = (sigma_T - sigma_S) / (sigma_T + sigma_S)
    WTG = (sigma_T - sigma_G) / (sigma_T + sigma_G)

    mapping = _omega(z_ - cell_z_mid)
    mapping += (WTS * _omega(z_ + cell_z_mid - 2 * h) +
                WTG * _omega(z_ + cell_z_mid))

    n = np.arange(1, steps)
    a = (WTS * WTG)**n[:, None] * (
        WTS * _omega(z_ + cell_z_mid - 2 * (n[:, None] + 1) * h) +
        WTG * _omega(z_ + cell_z_mid + 2 * n[:, None] * h) +
        _omega(z_ - cell_z_mid + 2 * n[:, None] * h) +
        _omega(z_ - cell_z_mid - 2 * n[:, None] * h))
    mapping += np.sum(a, axis=0)
    mapping *= 1 / (4 * np.pi * sigma_T)

    return mapping


def calc_lfp_linesource_moi(cell_x, cell_y, cell_z,
                            x, y, z,
                            sigma_T, sigma_S, sigma_G,
                            steps, h, r_limit, atol=1e-8, **kwargs):
    """Calculate extracellular potentials using the line-source
    equation on all compartments for in vitro Microelectrode Array (MEA) slices

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
    sigma_T: float
        extracellular conductivity in tissue slice
    sigma_G: float
        Conductivity of MEA glass electrode plane.
        Should normally be zero for MEA set up, and for this method,
        only zero valued sigma_G is supported.
    sigma_S: float
        Conductivity of saline bath that tissue slice is immersed in
    steps: int
        Number of steps to average over the in technically infinite sum
    h: float
        Slice thickness in um.
    r_limit: np.ndarray
        minimum distance to source current for each compartment
    atol: float
        numerical (absolute) tolerance for near-zero comparisons.
        Should be a small number, and should normally not need to be changed.
    """

    cell_z = cell_z.copy()  # Copy to safely shift coordinate system if needed
    z_ = z

    if "z_shift" in kwargs and kwargs["z_shift"] is not None:
        # Shifting coordinate system before calculations
        z_ -= kwargs["z_shift"]
        cell_z -= kwargs["z_shift"]

    if np.abs(z_) > atol:
        raise RuntimeError("This method can only handle electrodes "
                           "at the MEA plane z=0")
    if np.abs(sigma_G) > atol:
        raise RuntimeError("This method can only handle sigma_G=0, i.e.,"
                           "a non-conducting MEA glass electrode plane.")

    xstart = cell_x[:, 0]
    xend = cell_x[:, -1]
    ystart = cell_y[:, 0]
    yend = cell_y[:, -1]
    zstart = cell_z[:, 0]
    zend = cell_z[:, -1]
    x0, y0, z0 = cell_x[:, 0], cell_y[:, 0], cell_z[:, 0]
    x1, y1, z1 = cell_x[:, -1], cell_y[:, -1], cell_z[:, -1]

    pos = np.array([x, y, z_])
    rs, _ = return_dist_from_segments(xstart, ystart, zstart,
                                      xend, yend, zend, pos)
    z0_ = z0.copy()
    # avoid denominator approaching 0
    inds = rs < r_limit
    z0_[inds] = r_limit[inds]

    ds = _deltaS_calc(xstart, xend, ystart, yend, zstart, zend)
    factor_a = ds * ds
    dx = x1 - x0
    dy = y1 - y0
    dz = z1 - z0
    a_x = x - x0
    a_y = y - y0
    W = (sigma_T - sigma_S) / (sigma_T + sigma_S)
    num = np.zeros(factor_a.shape)
    den = np.zeros(factor_a.shape)

    def _omega(a_z):
        """ See Rottman integration formula 46) page 137 for explanation """
        factor_b = - a_x * dx - a_y * dy - a_z * dz
        factor_c = a_x * a_x + a_y * a_y + a_z * a_z
        b_2_ac = factor_b * factor_b - factor_a * factor_c

        case1_idxs = np.abs(b_2_ac) <= atol
        case2_idxs = np.abs(b_2_ac) > atol

        num[case1_idxs] = factor_a[case1_idxs] + factor_b[case1_idxs]
        den[case1_idxs] = factor_b[case1_idxs]

        num[case2_idxs] = (factor_a[case2_idxs] + factor_b[case2_idxs]
                           + ds[case2_idxs]
                           * np.sqrt(factor_a[case2_idxs]
                                     + 2 * factor_b[case2_idxs]
                                     + factor_c[case2_idxs])
                           )
        den[case2_idxs] = (factor_b[case2_idxs] +
                           ds[case2_idxs] * np.sqrt(factor_c[case2_idxs]))
        return np.log(num / den)

    mapping = _omega(-z0_)
    n = 1
    while n < steps:
        mapping += W**n * (_omega(2 * n * h - z0_) + _omega(-2 * n * h - z0_))
        n += 1

    mapping *= 2 / (4 * np.pi * sigma_T * ds)

    return mapping


def calc_lfp_root_as_point_moi(cell_x, cell_y, cell_z,
                               x, y, z,
                               sigma_T, sigma_S, sigma_G,
                               steps, h, r_limit, atol=1e-8, **kwargs):
    """Calculate extracellular potentials for in vitro
    Microelectrode Array (MEA) slices, where root (compartment zero) is
    treated as a point source, and all other compartments as line sources.

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
    sigma_T: float
        extracellular conductivity in tissue slice
    sigma_G: float
        Conductivity of MEA glass electrode plane.
        Should normally be zero for MEA set up, and for this method,
        only zero valued sigma_G is supported.
    sigma_S: float
        Conductivity of saline bath that tissue slice is immersed in
    steps: int
        Number of steps to average over the in technically infinite sum
    h: float
        Slice thickness in um.
    r_limit: np.ndarray
        minimum distance to source current for each compartment
    atol: float
        numerical (absolute) tolerance for near-zero comparisons.
        Should be a small number, and should normally not need to be changed.
    """

    cell_z = cell_z.copy()  # Copy to safely shift coordinate system if needed
    z_ = z

    if "z_shift" in kwargs and kwargs["z_shift"] is not None:
        # Shifting coordinate system before calculations
        z_ -= kwargs["z_shift"]
        cell_z -= kwargs["z_shift"]

    if np.abs(z_) > atol:
        raise RuntimeError("This method can only handle electrodes "
                           "at the MEA plane z=0")
    if np.abs(sigma_G) > atol:
        raise RuntimeError("This method can only handle sigma_G=0, i.e.,"
                           "a non-conducting MEA glass electrode plane.")

    xstart = cell_x[:, 0]
    xend = cell_x[:, -1]
    ystart = cell_y[:, 0]
    yend = cell_y[:, -1]
    zstart = cell_z[:, 0]
    zend = cell_z[:, -1]
    x0, y0, z0 = cell_x[:, 0], cell_y[:, 0], cell_z[:, 0]
    x1, y1, z1 = cell_x[:, -1], cell_y[:, -1], cell_z[:, -1]

    pos = np.array([x, y, z_])
    rs, _ = return_dist_from_segments(xstart, ystart, zstart,
                                      xend, yend, zend, pos)
    z0_ = np.array(z0)
    if np.any(rs < r_limit):
        z0_[rs < r_limit] = r_limit

    ds = _deltaS_calc(xstart, xend, ystart, yend, zstart, zend)
    factor_a = ds * ds
    dx = x1 - x0
    dy = y1 - y0
    dz = z1 - z0
    a_x = x - x0
    a_y = y - y0
    W = (sigma_T - sigma_S) / (sigma_T + sigma_S)
    num = np.zeros(factor_a.shape)
    den = np.zeros(factor_a.shape)

    def _omega(a_z):
        """ See Rottman integration formula 46) page 137 for explanation """
        factor_b = - a_x * dx - a_y * dy - a_z * dz
        factor_c = a_x * a_x + a_y * a_y + a_z * a_z
        b_2_ac = factor_b * factor_b - factor_a * factor_c

        case1_idxs = np.abs(b_2_ac) <= atol
        case2_idxs = np.abs(b_2_ac) > atol

        num[case1_idxs] = factor_a[case1_idxs] + factor_b[case1_idxs]
        den[case1_idxs] = factor_b[case1_idxs]

        num[case2_idxs] = (factor_a[case2_idxs] + factor_b[case2_idxs] +
                           + ds[case2_idxs]
                           * np.sqrt(factor_a[case2_idxs]
                                     + 2 * factor_b[case2_idxs]
                                     + factor_c[case2_idxs])
                           )
        den[case2_idxs] = (factor_b[case2_idxs] +
                           ds[case2_idxs] * np.sqrt(factor_c[case2_idxs]))
        return np.log(num / den)

    mapping = _omega(-z0_)
    n = 1
    while n < steps:
        mapping += W**n * (_omega(2 * n * h - z0) + _omega(-2 * n * h - z0))
        n += 1

    mapping *= 2 / (4 * np.pi * sigma_T * ds)

    # NOW DOING root

    # get compartment indices for root segment (to be treated as point
    # sources)
    rootinds = np.array([0])

    dx2 = (x - cell_x[rootinds, :].mean(axis=-1))**2
    dy2 = (y - cell_y[rootinds, :].mean(axis=-1))**2
    dz2 = (z_ - cell_z[rootinds, :].mean(axis=-1))**2

    dL2 = dx2 + dy2
    # avoid denominator approaching 0
    inds = (dL2 + dz2) < (r_limit[rootinds] * r_limit[rootinds])
    dL2[inds] = r_limit[rootinds][inds] * r_limit[rootinds][inds] - dz2[inds]

    def _omega(dz_):
        return 1 / np.sqrt(dL2 + dz_ * dz_)

    mapping[rootinds] = _omega(z_ - cell_z[rootinds, :].mean(axis=-1))
    mapping[rootinds] += (W * _omega(cell_z[rootinds, :].mean(axis=-1) - 2 * h
                                     )
                          + _omega(cell_z[rootinds, :].mean(axis=-1)))

    n = np.arange(1, steps)
    a = W**n[:, None] * (W * _omega(cell_z[rootinds, :].mean(axis=-1)
                         - 2 * (n[:, None] + 1) * h)
                         + 2 * _omega(cell_z[rootinds, :].mean(axis=-1)
                         + 2 * n[:, None] * h)
                         + _omega(cell_z[rootinds, :].mean(axis=-1)
                         - 2 * n[:, None] * h))
    mapping[rootinds] += np.sum(a, axis=0)
    mapping[rootinds] *= 1 / (4 * np.pi * sigma_T)

    return mapping
