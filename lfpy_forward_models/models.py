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


class LinearModel(object):
    '''
    Base LinearModel class skeleton that defines a 2D linear response
    matrix `M` between transmembrane currents `I` [nA] of a multicompartment
    neuron model and some measurement `Y` as

    .. math:: Y = MI

    LinearModel only creates a mapping that returns the currents themselves.
    The class is suitable as a base class for other linear model
    implementations, see for example class CurrentDipoleMoment

    Parameters
    ----------
    cell: object
        CellGeometry or similar instance.
    '''
    def __init__(self, cell):
        self.cell = cell

    def get_response_matrix(self):
        '''
        Get linear response matrix

        Returns
        -------
        response_matrix: ndarray
            shape (n_seg, n_seg) ndarray
        '''
        return np.eye(self.cell.totnsegs)


class CurrentDipoleMoment(LinearModel):
    '''
    LinearModel subclass that defines a 2D linear response matrix `M` between
    transmembrane current array `I` [nA] of a multicompartment neuron model
    and the corresponding current dipole moment `P` [nA um] as

    .. math:: P = MI


    The current `I` is an ndarray of shape (n_seg, n_tsteps) with unit [nA],
    and the rows of `P` represent the x-, y- and z-components of the current
    diple moment for every time step.

    The current dipole moment can be used to compute distal measures of
    neural activity such as the EEG and MEG using
    LFPy.FourSphereVolumeConductor or LFPy.MEG, respectively

    Parameters
    ----------
    cell: object
        CellGeometry or similar instance.

    Examples
    --------
    Compute the current dipole moment of a 3-compartment neuron model:

    >>> import numpy as np
    >>> from lfpy_forward_models import CellGeometry, CurrentDipoleMoment
    >>> n_seg = 3
    >>> cell = CellGeometry(x=np.array([[0.]*2]*n_seg),
                            y=np.array([[0.]*2]*n_seg),
                            z=np.array([[1.*x, 1.*(x+1)]
                                        for x in range(n_seg)]),
                            diam=np.array([1.]*n_seg))
    >>> cdm = CurrentDipoleMoment(cell)
    >>> M = cdm.get_response_matrix()
    >>> imem = np.array([[-1., 1.],
                         [0., 0.],
                         [1., -1.]])
    >>> P = M@imem
    >>> P
    array([[ 0.,  0.],
           [ 0.,  0.],
           [ 2., -2.]])
    '''
    def __init__(self, cell):
        super().__init__(cell=cell)

    def get_response_matrix(self):
        '''
        Get linear response matrix

        Returns
        -------
        response_matrix: ndarray
            shape (3, n_seg) ndarray
        '''
        return np.stack([self.cell.x.mean(axis=-1),
                         self.cell.y.mean(axis=-1),
                         self.cell.z.mean(axis=-1)])
