#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2022 Computational Neuroscience Group, NMBU.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
"""
import abc
import numpy as np
from .cellgeometry import CellGeometry
from . import models


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
        c_ind = np.array([], dtype=int)  # tracks which comp. owns segment
        idx = 0
        for i, sec in enumerate(cell.all):
            xp = (self.arc3d[i] / self.arc3d[i][-1])
            seg_l = 1 / sec.nseg / 2
            for seg in sec:
                seg_x = np.array([seg.x - seg_l, seg.x + seg_l])
                ind = (xp > seg_x.min()) & (xp < seg_x.max())
                seg_x = np.sort(np.r_[seg_x, xp[ind]])

                # interpolate to get values for start-, pt3d-, and
                # end-point for search segment:
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
        import neuron

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
        c_ind = np.array([], dtype=int)  # tracks which comp. owns segment
        idx = 0
        for i, sec in enumerate(cell.allseclist):
            xp = (cell.arc3d[i] / cell.arc3d[i][-1])
            seg_l = 1 / sec.nseg / 2
            for seg in sec:
                seg_x = np.array([seg.x - seg_l, seg.x + seg_l])
                ind = (xp > seg_x.min()) & (xp < seg_x.max())
                seg_x = np.sort(np.r_[seg_x, xp[ind]])

                # interpolate to get values for start-, pt3d-, and
                # end-point for search segment:
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


class CurrentDipoleMoment(models.CurrentDipoleMoment):
    '''subclass of ``lfpykit.CurrentDipoleMoment`` modified for
    instances of ``lfpykit.special.CellGeometry*``.
    Each compartment may consist of several segments, and this implementation
    accounts for their contributions normalized by surface area, that is,
    we assume constant transmembrane current density per area across each
    compartment and constant current source density per area per segment.

    Parameters
    ----------
    cell: object
        ``special.CellGeometry*`` instance or similar.
    x: ndarray of floats
        x-position of measurement sites (µm)
    y: ndarray of floats
        y-position of measurement sites (µm)
    z: ndarray of floats
        z-position of measurement sites (µm)
    sigma: float > 0
        scalar extracellular conductivity (S/m)

    See also
    --------
    lfpykit.CurrentDipoleMoment
    '''

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_transformation_matrix(self):
        '''Get linear response matrix

        Returns
        -------
        response_matrix: ndarray
            shape (n_coords, n_compartments) ndarray
        '''
        M_tmp = super().get_transformation_matrix()
        n_compartments = np.unique(self.cell._compartment_index).size
        M = np.zeros((self.x.size, n_compartments))
        for i in range(n_compartments):
            inds = self.cell._compartment_index == i
            M[:, i] = M_tmp[:, inds] @ (self.cell.area[inds] /
                                        self.cell.area[inds].sum())
        return M


class LineSourcePotential(models.LineSourcePotential):
    '''subclass of ``lfpykit.LineSourcePotential`` modified for
    instances of ``lfpykit.special.CellGeometry*``.
    Each compartment may consist of several segments, and this implementation
    accounts for their contributions normalized by surface area, that is,
    we assume constant transmembrane current density per area across each
    compartment and constant current source density per unit length per segment
    (inherent in the line-source approximation).

    Parameters
    ----------
    cell: object
        ``special.CellGeometry*`` instance or similar.
    x: ndarray of floats
        x-position of measurement sites (µm)
    y: ndarray of floats
        y-position of measurement sites (µm)
    z: ndarray of floats
        z-position of measurement sites (µm)
    sigma: float > 0
        scalar extracellular conductivity (S/m)

    See also
    --------
    lfpykit.LineSourcePotential
    '''

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_transformation_matrix(self):
        '''Get linear response matrix

        Returns
        -------
        response_matrix: ndarray
            shape (n_coords, n_compartments) ndarray
        '''
        M_tmp = super().get_transformation_matrix()
        n_compartments = np.unique(self.cell._compartment_index).size
        M = np.zeros((self.x.size, n_compartments))
        for i in range(n_compartments):
            inds = self.cell._compartment_index == i
            M[:, i] = M_tmp[:, inds] @ (self.cell.area[inds] /
                                        self.cell.area[inds].sum())
        return M


class PointSourcePotential(models.PointSourcePotential):
    '''subclass of ``lfpykit.PointSourcePotential`` modified for
    instances of ``lfpykit.special.CellGeometry*``.
    Each compartment may consist of several segments, and this implementation
    accounts for their contributions normalized by surface area, that is,
    we assume constant transmembrane current density per area across each
    compartment and constant current source density per area per segment.

    Parameters
    ----------
    cell: object
        ``lfpykit.special.CellGeometry*`` instance or similar.
    x: ndarray of floats
        x-position of measurement sites (µm)
    y: ndarray of floats
        y-position of measurement sites (µm)
    z: ndarray of floats
        z-position of measurement sites (µm)
    sigma: float > 0
        scalar extracellular conductivity (S/m)

    See also
    --------
    lfpykit.PointSourcePotential
    '''

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_transformation_matrix(self):
        '''Get linear response matrix

        Returns
        -------
        response_matrix: ndarray
            shape (n_coords, n_compartments) ndarray
        '''
        M_tmp = super().get_transformation_matrix()
        n_compartments = np.unique(self.cell._compartment_index).size
        M = np.zeros((self.x.size, n_compartments))
        for i in range(n_compartments):
            inds = self.cell._compartment_index == i
            M[:, i] = M_tmp[:, inds] @ (self.cell.area[inds] /
                                        self.cell.area[inds].sum())
        return M
