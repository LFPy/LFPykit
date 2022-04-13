#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2020 Computational Neuroscience Group, NMBU.

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


class CellGeometry(object):
    '''
    Base class representing the geometry of multicompartment neuron
    models.

    Assumptions
        - each compartment is piecewise linear between their start
          and endpoints
        - each compartment has a constant diameter
        - the transmembrane current density is constant along the
          compartment axis

    Parameters
    ----------
    x: ndarray of floats
        shape (n_seg x 2) array of start- and end-point coordinates of
        each compartment along x-axis in units of (µm)
    y: ndarray
        shape (n_seg x 2) array of start- and end-point coordinates of
        each compartment along y-axis in units of (µm)
    z: ndarray
        shape (n_seg x 2) array of start- and end-point coordinates of
        each compartment along z-axis in units of (µm)
    d: ndarray
        shape (n_seg) or shape (n_seg x 2) array of compartment
        diameters in units of (µm). If the 2nd axis is equal to 2,
        conical frusta is assumed.

    Attributes
    ----------
    totnsegs: int
        number of compartments
    x: ndarray of floats
        shape (totnsegs x 2) array of start- and end-point coordinates of
        each compartment along x-axis in units of (µm)
    y: ndarray
        shape (totnsegs x 2) array of start- and end-point coordinates of
        each compartment along y-axis in units of (µm)
    z: ndarray
        shape (totnsegs x 2) array of start- and end-point coordinates of
        each compartment along z-axis in units of (µm)
    d: ndarray
        shape (totnsegs) array of compartment diameters in units of (µm)
    length: ndarray
        lenght of each compartment in units of um
    area: ndarray
        array of compartment surface areas in units of um^2
    '''
    def __init__(self, x, y, z, d):
        # check input
        assert np.all([type(x) is np.ndarray,
                       type(y) is np.ndarray,
                       type(z) is np.ndarray,
                       type(d) is np.ndarray]), \
            'x, y, z and d must be of type numpy.ndarray'
        assert x.ndim == y.ndim == z.ndim == 2, \
            'x, y and z must be of shape (n_seg x 2)'
        assert x.shape == y.shape == z.shape, \
            'x, y and z must all be the same shape'
        assert x.shape[1] == 2, \
            'the last axis of x, y and z must be of length 2'
        assert d.shape == x.shape or (d.ndim == 1 and d.size == x.shape[0]), \
            'd must either be 1-dimensional with size == n_seg ' + \
            'or 2-dimensional with d.shape == x.shape'

        # set attributes
        self.x = x
        self.y = y
        self.z = z
        self.d = d

        # derived attributes
        self.totnsegs = self.x.shape[0]

        self._set_length()
        self._set_area()

    def _set_length(self):
        self.length = np.sqrt(np.diff(self.x, axis=-1)**2 +
                              np.diff(self.y, axis=-1)**2 +
                              np.diff(self.z, axis=-1)**2).flatten()

    def _set_area(self):
        if self.d.ndim == 1:
            self.area = self.length * np.pi * self.d
        else:
            # Surface area of conical frusta
            # A = pi*(r1+r2)*sqrt((r1-r2)^2 + h^2)
            r = self.d / 2
            self.area = np.pi * r.sum(axis=-1) * \
                np.sqrt(np.diff(r, axis=-1).ravel()**2 + self.length**2)


class CellGeometryArbor(CellGeometry):
    '''
    Class inherited from  ``lfpykit.CellGeometry`` for easier forward-model
    predictions in Arbor that keeps track of arbor.segment information
    for each CV.

    Parameters
    ----------
    p: ``arbor.place_pwlin`` object
        3-d locations and cables in a morphology (cf. ``arbor.place_pwlin``)
    cables: ``list``
        ``list`` of corresponding ``arbor.cable`` objects where transmembrane
        currents are recorded (cf. ``arbor.cable_probe_total_current_cell``)

    See also
    --------
    lfpykit.CellGeometry
    '''

    def __init__(self, p, cables):
        x, y, z, d = [np.array([], dtype=float).reshape((0, 2))] * 4
        c_ind = np.array([], dtype=int)  # tracks which CV owns segment
        for i, m in enumerate(cables):
            segs = p.segments([m])
            for j, seg in enumerate(segs):
                x = np.row_stack([x, [seg.prox.x, seg.dist.x]])
                y = np.row_stack([y, [seg.prox.y, seg.dist.y]])
                z = np.row_stack([z, [seg.prox.z, seg.dist.z]])
                d = np.row_stack(
                    [d, [seg.prox.radius * 2, seg.dist.radius * 2]])
                c_ind = np.r_[c_ind, i]

        super().__init__(x=x, y=y, z=z, d=d)
        self._compartment_index = c_ind


class CellGeometryNeuron(CellGeometry):
    '''
    Class inherited from  ``lfpykit.CellGeometry`` for easier forward-model
    predictions from NEURON that keeps track of pt3d information
    for each compartment.

    Parameters
    ----------
    cell: ``Cell`` object

    See also
    --------
    lfpykit.CellGeometry
    '''
    def __init__(self, cell):
        self.x3d, self.y3d, self.z3d, self.diam3d, self.arc3d = \
                self._collect_pt3d(cell)

        x, y, z, d = [np.array([], dtype=float).reshape((0, 2))] * 4
        c_ind = np.array([], dtype=int)  # tracks which compartment owns segment
        idx = 0
        for i, sec in enumerate(cell.all):
            xp = (self.arc3d[i] / self.arc3d[i][-1])
            seg_l = 1 / sec.nseg / 2
            for seg in sec:
                seg_x = np.array([seg.x - seg_l, seg.x + seg_l])
                ind = (xp > seg_x.min()) & (xp < seg_x.max())
                seg_x = np.sort(np.r_[seg_x, xp[ind]])

                # interpolate to get values for start-, pt3d-, and end-point for search segment:
                x_tmp = np.interp(seg_x, xp, self.x3d[i])
                y_tmp = np.interp(seg_x, xp, self.y3d[i])
                z_tmp = np.interp(seg_x, xp, self.z3d[i])
                d_tmp = np.interp(seg_x, xp, self.diam3d[i])

                # store
                x = np.row_stack([x, np.c_[x_tmp[:-1], x_tmp[1:]]])
                y = np.row_stack([y, np.c_[y_tmp[:-1], y_tmp[1:]]])
                z = np.row_stack([z, np.c_[z_tmp[:-1], z_tmp[1:]]])
                d = np.row_stack([d, np.c_[d_tmp[:-1], d_tmp[1:]]])

                c_ind = np.r_[c_ind, np.ones(seg_x.size - 1) * idx]
                idx += 1

        super().__init__(x=x, y=y, z=z, d=d)
        self._compartment_index = c_ind

    def _collect_pt3d(self, cell):
        """collect the pt3d info, for each section

        Returns
        -------
        x3d: list of ndarray
            x-coordinates from h.x3d()
        y3d: list of ndarray
            y-coordinates from h.y3d()
        z3d: list of ndarray
            z-coordinates from h.z3d()
        diam3d: list of ndarray
            diameter from h.diam3d()
        arc3d: list of ndarray
            arclength from h.arc3d()
        """
        x3d, y3d, z3d, diam3d, arc3d = [], [], [], [], []

        for sec in cell.all:
            n3d = int(neuron.h.n3d(sec=sec))
            x3d_i, y3d_i, z3d_i, diam3d_i, arc3d_i = np.zeros((5, n3d))
            for i in range(n3d):
                x3d_i[i] = neuron.h.x3d(i, sec=sec)
                y3d_i[i] = neuron.h.y3d(i, sec=sec)
                z3d_i[i] = neuron.h.z3d(i, sec=sec)
                diam3d_i[i] = neuron.h.diam3d(i, sec=sec)
                arc3d_i[i] = neuron.h.arc3d(i, sec=sec)

            x3d.append(x3d_i)
            y3d.append(y3d_i)
            z3d.append(z3d_i)
            diam3d.append(diam3d_i)
            arc3d.append(arc3d_i)

        return x3d, y3d, z3d, diam3d, arc3d


class CellGeometryLFPyPt3d(CellGeometry):
    '''
    Class inherited from  ``lfpykit.CellGeometry`` for easier forward-model
    predictions in LFPy that keeps track of pt3d information
    for each compartment.

    Parameters
    ----------
    cell: ``LFPy.Cell`` object

    See also
    --------
    lfpykit.CellGeometry
    '''
    def __init__(self, cell):
        x, y, z, d = [np.array([], dtype=float).reshape((0, 2))] * 4
        c_ind = np.array([], dtype=int)  # tracks which compartment owns segment
        idx = 0
        for i, sec in enumerate(cell.allseclist):
            xp = (cell.arc3d[i] / cell.arc3d[i][-1])
            seg_l = 1 / sec.nseg / 2
            for seg in sec:
                seg_x = np.array([seg.x - seg_l, seg.x + seg_l])
                ind = (xp > seg_x.min()) & (xp < seg_x.max())
                seg_x = np.sort(np.r_[seg_x, xp[ind]])

                # interpolate to get values for start-, pt3d-, and end-point for search segment:
                x_tmp = np.interp(seg_x, xp, cell.x3d[i])
                y_tmp = np.interp(seg_x, xp, cell.y3d[i])
                z_tmp = np.interp(seg_x, xp, cell.z3d[i])
                d_tmp = np.interp(seg_x, xp, cell.diam3d[i])

                # store
                x = np.row_stack([x, np.c_[x_tmp[:-1], x_tmp[1:]]])
                y = np.row_stack([y, np.c_[y_tmp[:-1], y_tmp[1:]]])
                z = np.row_stack([z, np.c_[z_tmp[:-1], z_tmp[1:]]])
                d = np.row_stack([d, np.c_[d_tmp[:-1], d_tmp[1:]]])

                c_ind = np.r_[c_ind, np.ones(seg_x.size - 1) * idx]
                idx += 1

        super().__init__(x=x, y=y, z=z, d=d)
        self._compartment_index = c_ind
