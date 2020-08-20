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
    Base LinearModel class skeleton that defines a linear response matrix M
    between transmembrane currents I of a multicompartment neuron model and
    some measurement Y as Y=MI.

    LinearModel just returns a mapping that returns the current themselves

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
    LinearModel subclass that defines a linear response matrix M between
    transmembrane currents I of a multicompartment neuron model and the
    corresponding current dipole moment p as p = MI.

    The current dipole moment can be used to compute distal measures of
    neural activity such as the EEG and MEG

    Parameters
    ----------
    cell: object
        CellGeometry or similar instance.
    '''
    def __init__(self, cell):
        LinearModel.__init__(self, cell=cell)

    def get_response_matrix(self):
        '''
        Get linear response matrix

        Returns
        -------
        response_matrix: ndarray
            shape (3, n_seg) ndarray
        '''
        return np.stack([self.cell.xmid, self.cell.ymid, self.cell.zmid])
