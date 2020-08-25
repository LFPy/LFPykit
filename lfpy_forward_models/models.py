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
from . import lfpcalc


class LinearModel(object):
    '''
    Base LinearModel class skeleton that defines a 2D linear response
    matrix :math:`M` between transmembrane currents :math:`I` [nA] of a
    multicompartment neuron model and some measurement :math:`Y` as

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
    LinearModel subclass that defines a 2D linear response matrix :math:`M`
    between transmembrane current array :math:`I` [nA] of a multicompartment
    neuron model and the corresponding current dipole moment :math:`P` [nA um]
    as

    .. math:: P = MI


    The current :math:`I` is an ndarray of shape (n_seg, n_tsteps) with
    unit [nA], and the rows of :math:`P` represent the x-, y- and z-components
    of the current diple moment for every time step.

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
                            d=np.array([1.]*n_seg))
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


class PointSourcePotential(LinearModel):
    '''
    LinearModel subclass that defines a 2D linear response matrix :math:`M`
    between transmembrane current array :math:`I` [nA] of a multicompartment
    neuron model and the corresponding extracellular electric potential
    :math:`V_{ex}` [mV] as

    .. math:: V_{ex} = MI

    The current :math:`I` is an ndarray of shape (n_seg, n_tsteps) with
    unit [nA], and each row indexed by :math:`j` of :math:`V_{ex}` represents
    the electric potential at each measurement site for every time step.
    The elements of :math:`M` are computed as

    .. math:: M_{ji} = 1 / (4 \\pi \\sigma |r_i - r_j|)

    where :math:`\\sigma` is the electric conductivity of the extracellular
    medium, :math:`r_i` the midpoint coordinate of segment :math:`i`
    and :math:`r_j` the coordinate of measurement site :math:`j` [1, 2].

    Assumptions:
        - the extracellular conductivity :math:`\\sigma` is infinite,
          homogeneous, frequency independent (linear) and isotropic
        - each segment is treated as a point source located at the midpoint
          between its start and end point coordinate
        - each measurement site :math:`r_j = (x_j, y_j, z_j)` is treated
          as a point
        - :math:`|r_i - r_j| >= d_i / 2`, where `d_i` is the segment diameter.

    Parameters
    ----------
    cell: object
        CellGeometry or similar instance.
    x: ndarray of floats
        x-position of measurement sites [um]
    y: ndarray of floats
        y-position of measurement sites [um]
    z: ndarray of floats
        z-position of measurement sites [um]
    sigma: float > 0
        scalar extracellular conductivity [S/m]

    Examples
    --------
    Compute the current dipole moment of a 3-compartment neuron model:

    >>> import numpy as np
    >>> from lfpy_forward_models import CellGeometry, PointSourcePotential
    >>> n_seg = 3
    >>> cell = CellGeometry(x=np.array([[0.]*2]*n_seg),
                            y=np.array([[0.]*2]*n_seg),
                            z=np.array([[10.*x, 10.*(x+1)]
                                        for x in range(n_seg)]),
                            d=np.array([1.]*n_seg))
    >>> psp = PointSourcePotential(cell,
                                   x=np.ones(10)*10,
                                   y=np.zeros(10),
                                   z=np.arange(10)*10,
                                   sigma=0.3)
    >>> M = psp.get_response_matrix()
    >>> imem = np.array([[-1., 1.],
                         [0., 0.],
                         [1., -1.]])
    >>> V_ex = M @ imem
    >>> V_ex
    array([[-0.01387397,  0.01387397],
           [-0.00901154,  0.00901154],
           [ 0.00901154, -0.00901154],
           [ 0.01387397, -0.01387397],
           [ 0.00742668, -0.00742668],
           [ 0.00409718, -0.00409718],
           [ 0.00254212, -0.00254212],
           [ 0.00172082, -0.00172082],
           [ 0.00123933, -0.00123933],
           [ 0.00093413, -0.00093413]])


    References:
        [1] Linden H, Hagen E, Leski S, Norheim ES, Pettersen KH, Einevoll GT
            (2014) LFPy: a tool for biophysical simulation of extracellular
            potentials generated by detailed model neurons. Front.
            Neuroinform. 7:41. doi: 10.3389/fninf.2013.00041
        [2] Hagen E, Næss S, Ness TV and Einevoll GT (2018) Multimodal Modeling
            of Neural Network Activity: Computing LFP, ECoG, EEG, and MEG
            Signals With LFPy 2.0. Front. Neuroinform. 12:92.
            doi: 10.3389/fninf.2018.00092
    '''
    def __init__(self, cell, x, y, z, sigma=0.3):
        super().__init__(cell=cell)

        # check input
        try:
            assert(np.all([type(x) is np.ndarray,
                           type(x) is np.ndarray,
                           type(x) is np.ndarray]))
        except AssertionError as ae:
            raise ae('x, y and z must be of type numpy.ndarray')
        try:
            assert(x.ndim == y.ndim == z.ndim == 1)
        except AssertionError as ae:
            raise ae('x, y and z must be of shape (n_coords, )')
        try:
            assert(x.shape == y.shape == z.shape)
        except AssertionError as ae:
            raise ae('x, y and z must contain the same number of elements')
        try:
            assert(type(sigma) is float and sigma > 0)
        except AssertionError as ae:
            raise ae('sigma must be a float number greater than zero')

        # set attributes
        self.x = x
        self.y = y
        self.z = z
        self.sigma = sigma

    def get_response_matrix(self):
        '''
        Get linear response matrix

        Returns
        -------
        response_matrix: ndarray
            shape (n_coords, n_seg) ndarray
        '''
        M = np.empty((self.x.size, self.cell.totnsegs))
        for j in range(self.x.size):
            M[j, :] = lfpcalc.calc_lfp_pointsource(self.cell,
                                                   x=self.x[j],
                                                   y=self.y[j],
                                                   z=self.z[j],
                                                   sigma=self.sigma,
                                                   r_limit=self.cell.d / 2)
        return M


class LineSourcePotential(LinearModel):
    '''
    LinearModel subclass that defines a 2D linear response matrix :math:`M`
    between transmembrane current array :math:`I` [nA] of a multicompartment
    neuron model and the corresponding extracellular electric potential
    :math:`V_ex` [mV] as

    .. math:: V_{ex} = MI

    The current :math:`I` is an ndarray of shape (n_seg, n_tsteps) with
    unit [nA], and each row indexed by :math:`j` of :math:`V_{ex}` represents
    the electric potential at each measurement site for every time step.
    The elements of :math:`M` are computed as

    .. math:: M_{ji} = \\frac{1}{ 4 \\pi \\sigma L_i } \\log
        \\left|
        \\frac{\\sqrt{h_{ji}^2+r_{ji}^2}-h_{ji}
               }{
               \\sqrt{l_{ji}^2+r_{ji}^2}-l_{ji}}
        \\right|


    Segment length is denoted :math:`L_i`, perpendicular distance from the
    electrode point contact to the axis of the line segment is denoted
    :math:`r_{ji}`, longitudinal distance measured from the start of the
    segment is denoted :math:`h_{ji}`, and longitudinal distance from the other
    end of the segment is denoted :math:`l_{ji}= L_i + h_{ji}` [1, 2].

    Assumptions:
        - the extracellular conductivity :math:`\\sigma` is infinite,
          homogeneous, frequency independent (linear) and isotropic
        - each segment is treated as a straigh line source with homogeneous
          current density between its start and end point coordinate
        - each measurement site :math:`r_j = (x_j, y_j, z_j)` is treated
          as a point
        - The minimum distance to a line source is set equal to segment radius.

    Parameters
    ----------
    cell: object
        CellGeometry or similar instance.
    x: ndarray of floats
        x-position of measurement sites [um]
    y: ndarray of floats
        y-position of measurement sites [um]
    z: ndarray of floats
        z-position of measurement sites [um]
    sigma: float > 0
        scalar extracellular conductivity [S/m]

    Examples
    --------
    Compute the current dipole moment of a 3-compartment neuron model:

    >>> import numpy as np
    >>> from lfpy_forward_models import CellGeometry, LineSourcePotential
    >>> n_seg = 3
    >>> cell = CellGeometry(x=np.array([[0.]*2]*n_seg),
                            y=np.array([[0.]*2]*n_seg),
                            z=np.array([[10.*x, 10.*(x+1)]
                                        for x in range(n_seg)]),
                            d=np.array([1.]*n_seg))
    >>> lsp = LineSourcePotential(cell,
                                  x=np.ones(10)*10,
                                  y=np.zeros(10),
                                  z=np.arange(10)*10,
                                  sigma=0.3)
    >>> M = lsp.get_response_matrix()
    >>> imem = np.array([[-1., 1.],
                         [0., 0.],
                         [1., -1.]])
    >>> V_ex = M @ imem
    >>> V_ex
    array([[-0.01343699,  0.01343699],
           [-0.0084647 ,  0.0084647 ],
           [ 0.0084647 , -0.0084647 ],
           [ 0.01343699, -0.01343699],
           [ 0.00758627, -0.00758627],
           [ 0.00416681, -0.00416681],
           [ 0.002571  , -0.002571  ],
           [ 0.00173439, -0.00173439],
           [ 0.00124645, -0.00124645],
           [ 0.0009382 , -0.0009382 ]])

    References:
        [1] Linden H, Hagen E, Leski S, Norheim ES, Pettersen KH, Einevoll GT
            (2014) LFPy: a tool for biophysical simulation of extracellular
            potentials generated by detailed model neurons. Front.
            Neuroinform. 7:41. doi: 10.3389/fninf.2013.00041
        [2] Hagen E, Næss S, Ness TV and Einevoll GT (2018) Multimodal Modeling
            of Neural Network Activity: Computing LFP, ECoG, EEG, and MEG
            Signals With LFPy 2.0. Front. Neuroinform. 12:92.
            doi: 10.3389/fninf.2018.00092
    '''
    def __init__(self, cell, x, y, z, sigma=0.3):
        super().__init__(cell=cell)

        # check input
        try:
            assert(np.all([type(x) is np.ndarray,
                           type(x) is np.ndarray,
                           type(x) is np.ndarray]))
        except AssertionError as ae:
            raise ae('x, y and z must be of type numpy.ndarray')
        try:
            assert(x.ndim == y.ndim == z.ndim == 1)
        except AssertionError as ae:
            raise ae('x, y and z must be of shape (n_coords, )')
        try:
            assert(x.shape == y.shape == z.shape)
        except AssertionError as ae:
            raise ae('x, y and z must contain the same number of elements')
        try:
            assert(type(sigma) is float and sigma > 0)
        except AssertionError as ae:
            raise ae('sigma must be a float number greater than zero')

        # set attributes
        self.x = x
        self.y = y
        self.z = z
        self.sigma = sigma

    def get_response_matrix(self):
        '''
        Get linear response matrix

        Returns
        -------
        response_matrix: ndarray
            shape (n_coords, n_seg) ndarray
        '''
        M = np.empty((self.x.size, self.cell.totnsegs))
        for j in range(self.x.size):
            M[j, :] = lfpcalc.calc_lfp_linesource(self.cell,
                                                  x=self.x[j],
                                                  y=self.y[j],
                                                  z=self.z[j],
                                                  sigma=self.sigma,
                                                  r_limit=self.cell.d / 2)
        return M
