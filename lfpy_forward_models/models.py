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

import sys
from copy import deepcopy
import numpy as np
from . import lfpcalc
import MEAutility as mu


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
        CellGeometry instance or similar.
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
        CellGeometry instance or similar.

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
    and :math:`r_j` the coordinate of measurement site :math:`j` [1]_, [2]_.

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
        CellGeometry instance or similar.
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


    References
    ----------
    .. [1] Linden H, Hagen E, Leski S, Norheim ES, Pettersen KH, Einevoll GT
       (2014) LFPy: a tool for biophysical simulation of extracellular
       potentials generated by detailed model neurons. Front.
       Neuroinform. 7:41. doi: 10.3389/fninf.2013.00041
    .. [2] Hagen E, Næss S, Ness TV and Einevoll GT (2018) Multimodal Modeling
       of Neural Network Activity: Computing LFP, ECoG, EEG, and MEG
       Signals With LFPy 2.0. Front. Neuroinform. 12:92.
       doi: 10.3389/fninf.2018.00092
    '''

    def __init__(self, cell, x, y, z, sigma=0.3):
        super().__init__(cell=cell)

        # check input
        try:
            assert(np.all([isinstance(x, np.ndarray),
                           isinstance(y, np.ndarray),
                           isinstance(z, np.ndarray)]))
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
            assert(isinstance(sigma, float) and sigma > 0)
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
    end of the segment is denoted :math:`l_{ji}= L_i + h_{ji}` [1]_, [2]_.

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
        CellGeometry instance or similar.
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

    References
    ----------
    .. [1] Linden H, Hagen E, Leski S, Norheim ES, Pettersen KH, Einevoll GT
       (2014) LFPy: a tool for biophysical simulation of extracellular
       potentials generated by detailed model neurons. Front.
       Neuroinform. 7:41. doi: 10.3389/fninf.2013.00041
    .. [2] Hagen E, Næss S, Ness TV and Einevoll GT (2018) Multimodal Modeling
       of Neural Network Activity: Computing LFP, ECoG, EEG, and MEG
       Signals With LFPy 2.0. Front. Neuroinform. 12:92.
       doi: 10.3389/fninf.2018.00092
    '''

    def __init__(self, cell, x, y, z, sigma=0.3):
        super().__init__(cell=cell)

        # check input
        try:
            assert(np.all([isinstance(x, np.ndarray),
                           isinstance(y, np.ndarray),
                           isinstance(z, np.ndarray)]))
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
            assert(isinstance(sigma, float) and sigma > 0)
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


class RecExtElectrode(LinearModel):
    """class RecExtElectrode

    Main class that represents an extracellular electric recording devices such
    as a laminar probe.

    Parameters
    ----------
    cell: object
        CellGeometry instance or similar.
    sigma: float or list/ndarray of floats
        extracellular conductivity in units of [S/m]. A scalar value implies an
        isotropic extracellular conductivity. If a length 3 list or array of
        floats is provided, these values corresponds to an anisotropic
        conductor with conductivities [sigma_x, sigma_y, sigma_z] accordingly.
    probe: MEAutility MEA object or None
        MEAutility probe object
    x, y, z: ndarray
        coordinates or same length arrays of coordinates in units of [um].
    N: None or list of lists
        Normal vectors [x, y, z] of each circular electrode contact surface,
        default None
    r: float
        radius of each contact surface, default None [um]
    n: int
        if N is not None and r > 0, the number of discrete points used to
        compute the n-point average potential on each circular contact point.
    contact_shape: str
        'circle'/'square' (default 'circle') defines the contact point shape
        If 'circle' r is the radius, if 'square' r is the side length
    method: str
        switch between the assumption of 'linesource', 'pointsource',
        'root_as_point' to represent each compartment when computing
        extracellular potentials
    verbose: bool
        Flag for verbose output, i.e., print more information
    seedvalue: int
        random seed when finding random position on contact with r > 0

    Examples
    --------

    Mock cell geometry and transmembrane currents:

    >>> import numpy as np
    >>> from lfpy_forward_models import CellGeometry, RecExtElectrode
    >>> # cell geometry with three segments [um]
    >>> cell = CellGeometry(x=np.array([[0, 0], [0, 0], [0, 0]]),
    >>>                     y=np.array([[0, 0], [0, 0], [0, 0]]),
    >>>                     z=np.array([[0, 10], [10, 20], [20, 30]]),
    >>>                     d=np.array([1, 1, 1]))
    >>> # transmembrane currents, three time steps [nA]
    >>> I_m = np.array([[0., -1., 1.], [-1., 1., 0.], [1., 0., -1.]])
    >>> # electrode locations [um]
    >>> r = np.array([[28.24653166, 8.97563241, 18.9492774, 3.47296614,
    >>>                1.20517729, 9.59849603, 21.91956616, 29.84686727,
    >>>                4.41045505, 3.61146625],
    >>>               [24.4954352, 24.04977922, 22.41262238, 10.09702942,
    >>>                3.28610789, 23.50277637, 8.14044367, 4.46909208,
    >>>                10.93270117, 24.94698813],
    >>>               [19.16644585, 15.20196335, 18.08924828, 24.22864702,
    >>>                5.85216751, 14.8231048, 24.72666694, 17.77573431,
    >>>                29.34508292, 9.28381892]])
    >>> # instantiate electrode, get linear response matrix
    >>> el = RecExtElectrode(cell=cell, x=r[0, ], y=r[1, ], z=r[2, ],
    >>>                      sigma=0.3,
    >>>                      method='pointsource')
    >>> M = el.get_response_matrix()
    >>> # compute extracellular potential
    >>> M @ I_m
    array([[-4.11657148e-05,  4.16621950e-04, -3.75456235e-04],
           [-6.79014892e-04,  7.30256301e-04, -5.12414088e-05],
           [-1.90930536e-04,  7.34007655e-04, -5.43077119e-04],
           [ 5.98270144e-03,  6.73490846e-03, -1.27176099e-02],
           [-1.34547752e-02, -4.65520036e-02,  6.00067788e-02],
           [-7.49957880e-04,  7.03763787e-04,  4.61940938e-05],
           [ 8.69330232e-04,  1.80346156e-03, -2.67279180e-03],
           [-2.04546513e-04,  6.58419628e-04, -4.53873115e-04],
           [ 6.82640209e-03,  4.47953560e-03, -1.13059377e-02],
           [-1.33289553e-03, -1.11818140e-04,  1.44471367e-03]])


    Compute extracellular potentials after simulating and storage of
    transmembrane currents with the LFPy.Cell class:

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import LFPy
    >>> from lfpy_forward_models import CellGeometry, RecExtElectrode
    >>>
    >>> cellParameters = {
    >>>     'morphology' : 'examples/morphologies/L5_Mainen96_LFPy.hoc',
    >>>     'v_init' : -65,                         # initial voltage
    >>>     'cm' : 1.0,                             # membrane capacitance
    >>>     'Ra' : 150,                             # axial resistivity
    >>>     'passive' : True,                       # insert passive channels
    >>>     'passive_parameters' : {"g_pas":1./3E4,
    >>>                             "e_pas":-65}, # passive params
    >>>     'dt' : 2**-4,                         # simulation time res
    >>>     'tstart' : 0.,                        # start t of simulation
    >>>     'tstop' : 50.,                        # end t of simulation
    >>> }
    >>> cell = LFPy.Cell(**cellParameters)
    >>>
    >>> synapseParameters = {
    >>>     'idx' : cell.get_closest_idx(x=0, y=0, z=800), # segment
    >>>     'e' : 0,                                # reversal potential
    >>>     'syntype' : 'ExpSyn',                   # synapse type
    >>>     'tau' : 2,                              # syn. time constant
    >>>     'weight' : 0.01,                        # syn. weight
    >>>     'record_current' : True                 # syn. current record
    >>> }
    >>> synapse = LFPy.Synapse(cell, **synapseParameters)
    >>> synapse.set_spike_times(np.array([10., 15., 20., 25.]))
    >>>
    >>> cell.simulate(rec_imem=True)
    >>>
    >>> N = np.empty((16, 3))
    >>> for i in range(N.shape[0]): N[i,] = [1, 0, 0] # normal vectors
    >>> electrodeParameters = {         # parameters for RecExtElectrode class
    >>>     'sigma' : 0.3,              # Extracellular potential
    >>>     'x' : np.zeros(16)+25,      # Coordinates of electrode contacts
    >>>     'y' : np.zeros(16),
    >>>     'z' : np.linspace(-500,1000,16),
    >>>     'n' : 20,
    >>>     'r' : 10,
    >>>     'N' : N,
    >>> }
    >>> cell_geometry = CellGeometry(
    >>>     x=np.c_[cell.xstart, cell.xend],
    >>>     y=np.c_[cell.ystart, cell.yend],
    >>>     z=np.c_[cell.zstart, cell.zend],
    >>>     d=cell.diam)
    >>> electrode = RecExtElectrode(cell_geometry, **electrodeParameters)
    >>> M = electrode.get_response_matrix()
    >>> V_ex = M @ cell.imem
    >>> plt.matshow(V_ex)
    >>> plt.colorbar()
    >>> plt.axis('tight')
    >>> plt.show()

    Compute extracellular potentials during simulation (recommended):

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import LFPy
    >>> from lfpy_forward_models import CellGeometry, RecExtElectrode
    >>>
    >>> cellParameters = {
    >>>     'morphology' : 'examples/morphologies/L5_Mainen96_LFPy.hoc',
    >>>     'v_init' : -65,                         # initial voltage
    >>>     'cm' : 1.0,                             # membrane capacitance
    >>>     'Ra' : 150,                             # axial resistivity
    >>>     'passive' : True,                       # insert passive channels
    >>>     'passive_parameters' : {"g_pas":1./3E4,
    >>>                             "e_pas":-65}, # passive params
    >>>     'dt' : 2**-4,                         # simulation time res
    >>>     'tstart' : 0.,                        # start t of simulation
    >>>     'tstop' : 50.,                        # end t of simulation
    >>> }
    >>> cell = LFPy.Cell(**cellParameters)
    >>>
    >>> synapseParameters = {
    >>>     'idx' : cell.get_closest_idx(x=0, y=0, z=800), # compartment
    >>>     'e' : 0,                                # reversal potential
    >>>     'syntype' : 'ExpSyn',                   # synapse type
    >>>     'tau' : 2,                              # syn. time constant
    >>>     'weight' : 0.01,                        # syn. weight
    >>>     'record_current' : True                 # syn. current record
    >>> }
    >>> synapse = LFPy.Synapse(cell, **synapseParameters)
    >>> synapse.set_spike_times(np.array([10., 15., 20., 25.]))
    >>>
    >>> N = np.empty((16, 3))
    >>> for i in range(N.shape[0]): N[i,] = [1, 0, 0] #normal vec. of contacts
    >>> electrodeParameters = {         # parameters for RecExtElectrode class
    >>>     'sigma' : 0.3,              # Extracellular potential
    >>>     'x' : np.zeros(16)+25,      # Coordinates of electrode contacts
    >>>     'y' : np.zeros(16),
    >>>     'z' : np.linspace(-500,1000,16),
    >>>     'n' : 20,
    >>>     'r' : 10,
    >>>     'N' : N,
    >>> }
    >>> cell_geometry = CellGeometry(
    >>>     x=np.c_[cell.xstart, cell.xend],
    >>>     y=np.c_[cell.ystart, cell.yend],
    >>>     z=np.c_[cell.zstart, cell.zend],
    >>>     d=cell.diam)
    >>> electrode = RecExtElectrode(cell_geometry, **electrodeParameters)
    >>> M = electrode.get_response_matrix()
    >>> cell.simulate(dotprodcoeffs=[M])
    >>> V_ex = cell.dotprodresults[0]
    >>> plt.matshow(V_ex)
    >>> plt.colorbar()
    >>> plt.axis('tight')
    >>> plt.show()

    Use MEAutility to to handle probes

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import MEAutility as mu
    >>> import LFPy
    >>> from lfpy_forward_models import CellGeometry, RecExtElectrode
    >>>
    >>> cellParameters = {
    >>>     'morphology' : 'examples/morphologies/L5_Mainen96_LFPy.hoc',
    >>>     'v_init' : -65,                         # initial voltage
    >>>     'cm' : 1.0,                             # membrane capacitance
    >>>     'Ra' : 150,                             # axial resistivity
    >>>     'passive' : True,                       # insert passive channels
    >>>     'passive_parameters' : {"g_pas":1./3E4,
    >>>                             "e_pas":-65}, # passive params
    >>>     'dt' : 2**-4,                         # simulation time res
    >>>     'tstart' : 0.,                        # start t of simulation
    >>>     'tstop' : 50.,                        # end t of simulation
    >>> }
    >>> cell = LFPy.Cell(**cellParameters)
    >>>
    >>> synapseParameters = {
    >>>     'idx' : cell.get_closest_idx(x=0, y=0, z=800), # compartment
    >>>     'e' : 0,                                # reversal potential
    >>>     'syntype' : 'ExpSyn',                   # synapse type
    >>>     'tau' : 2,                              # syn. time constant
    >>>     'weight' : 0.01,                        # syn. weight
    >>>     'record_current' : True                 # syn. current record
    >>> }
    >>> synapse = LFPy.Synapse(cell, **synapseParameters)
    >>> synapse.set_spike_times(np.array([10., 15., 20., 25.]))
    >>>
    >>> cell.simulate(rec_imem=True)
    >>>
    >>> probe = mu.return_mea('Neuropixels-128')
    >>> cell_geometry = CellGeometry(
    >>>     x=np.c_[cell.xstart, cell.xend],
    >>>     y=np.c_[cell.ystart, cell.yend],
    >>>     z=np.c_[cell.zstart, cell.zend],
    >>>     d=cell.diam)
    >>> electrode = RecExtElectrode(cell_geometry, probe=probe)
    >>> V_ex = electrode.get_response_matrix() @ cell.imem
    >>> mu.plot_mea_recording(V_ex, probe)
    >>> plt.axis('tight')
    >>> plt.show()
    """

    def __init__(self, cell, sigma=0.3, probe=None,
                 x=None, y=None, z=None,
                 N=None, r=None, n=None, contact_shape='circle',
                 method='linesource',
                 verbose=False,
                 seedvalue=None, **kwargs):
        """Initialize RecExtElectrode class"""
        super().__init__(cell=cell)

        self.sigma = sigma
        if type(sigma) in [list, np.ndarray]:
            self.sigma = np.array(sigma)
            if not self.sigma.shape == (3,):
                raise ValueError("Conductivity, sigma, should be float "
                                 "or ndarray of length 3: "
                                 "[sigma_x, sigma_y, sigma_z]")
            self.anisotropic = True
        else:
            self.sigma = sigma
            self.anisotropic = False

        if probe is None:
            assert np.all([arg is not None] for arg in [x, y, z]), \
                "instance requires either 'probe' or 'x', 'y', and 'z'"

            if type(x) in [float, int]:
                self.x = np.array([x])
            else:
                self.x = np.array(x).flatten()
            if type(y) in [float, int]:
                self.y = np.array([y])
            else:
                self.y = np.array(y).flatten()
            if type(z) in [float, int]:
                self.z = np.array([z])
            else:
                self.z = np.array(z).flatten()
            try:
                assert ((self.x.size == self.y.size) and
                        (self.x.size == self.z.size))
            except AssertionError as ae:
                raise ae("The number of elements in [x, y, z] must be equal")

            if N is not None:
                if not isinstance(N, np.ndarray):
                    try:
                        N = np.array(N)
                    except TypeError as te:
                        print('Keyword argument N could not be converted to a '
                              'numpy.ndarray of shape (n_contacts, 3)')
                        print(sys.exc_info()[0])
                        raise te
                if N.shape[-1] == 3:
                    self.N = N
                else:
                    self.N = N.T
                    if N.shape[-1] != 3:
                        raise Exception('N.shape must be (n_contacts, 1, 3)!')
            else:
                self.N = N

            self.r = r
            self.n = n

            if contact_shape is None:
                self.contact_shape = 'circle'
            elif contact_shape in ['circle', 'square', 'rect']:
                self.contact_shape = contact_shape
            else:
                raise ValueError('The contact_shape argument must be either: '
                                 'None, \'circle\', \'square\', \'rect\'')
            if self.contact_shape == 'rect':
                assert len(np.array(self.r)) == 2, \
                    "For 'rect' shape, 'r' indicates rectangle side length"

            positions = np.array([self.x, self.y, self.z]).T
            probe_info = {'pos': positions,
                          'description': 'custom',
                          'size': self.r,
                          'shape': self.contact_shape,
                          'type': 'wire',
                          'center': False}  # add mea type
            self.probe = mu.MEA(positions=positions, info=probe_info,
                                normal=self.N, sigma=self.sigma)
        else:
            assert isinstance(probe, mu.core.MEA), \
                "'probe' should be a MEAutility MEA object"
            self.probe = deepcopy(probe)
            self.x = probe.positions[:, 0]
            self.y = probe.positions[:, 1]
            self.z = probe.positions[:, 2]
            self.N = np.array([el.normal for el in self.probe.electrodes])
            self.r = self.probe.size
            self.contact_shape = self.probe.shape
            self.n = n

        self.method = method
        self.verbose = verbose
        self.seedvalue = seedvalue

        self.kwargs = kwargs

        # None-type some attributes created by the Cell class
        self.electrodecoeff = None
        self.circle = None
        self.offsets = None

        if method == 'root_as_point':
            if self.anisotropic:
                self.lfp_method = lfpcalc.calc_lfp_root_as_point_anisotropic
            else:
                self.lfp_method = lfpcalc.calc_lfp_root_as_point
        elif method == 'linesource':
            if self.anisotropic:
                self.lfp_method = lfpcalc.calc_lfp_linesource_anisotropic
            else:
                self.lfp_method = lfpcalc.calc_lfp_linesource
        elif method == 'pointsource':
            if self.anisotropic:
                self.lfp_method = lfpcalc.calc_lfp_pointsource_anisotropic
            else:
                self.lfp_method = lfpcalc.calc_lfp_pointsource
        else:
            raise ValueError("LFP method not recognized. "
                             "Should be 'root_as_point', 'linesource' "
                             "or 'pointsource'")

    def get_response_matrix(self):
        '''
        Get linear response matrix

        Returns
        -------
        response_matrix: ndarray
            shape (n_coords, n_seg) ndarray
        '''
        if self.n is not None and self.N is not None and self.r is not None:
            if self.n <= 1:
                raise ValueError("n = %i must be larger that 1" % self.n)
            else:
                pass

            M = self._lfp_el_pos_calc_dist()

            if self.verbose:
                print('calculations finished, %s, %s' % (str(self),
                                                         str(self.cell)))
        else:
            M = self._loop_over_contacts()
            if self.verbose:
                print('calculations finished, %s, %s' % (str(self),
                                                         str(self.cell)))
        # return mapping
        return M

    def _loop_over_contacts(self, **kwargs):
        """Loop over electrode contacts, and return LFPs across channels"""
        M = np.zeros((self.x.size, self.cell.x.shape[0]))
        for i in range(self.x.size):
            M[i, :] = self.lfp_method(self.cell,
                                      x=self.x[i],
                                      y=self.y[i],
                                      z=self.z[i],
                                      sigma=self.sigma,
                                      r_limit=self.cell.d / 2,
                                      **kwargs)
        return M

    def _lfp_el_pos_calc_dist(self, **kwargs):
        """
        Calc. of LFP over an n-point integral approximation over flat
        electrode surface: circle of radius r or square of side r. The
        locations of these n points on the electrode surface are random,
        within the given surface. """

        def loop_over_points(points):

            # loop over points on contact
            lfp_e = 0
            for j in range(self.n):
                tmp = self.lfp_method(self.cell,
                                      x=points[j, 0],
                                      y=points[j, 1],
                                      z=points[j, 2],
                                      r_limit=self.cell.d / 2,
                                      sigma=self.sigma,
                                      **kwargs
                                      )
                lfp_e += tmp

            return lfp_e / self.n

        # linear response matrix
        M = np.zeros((self.x.size, self.cell.x.shape[0]))

        # extract random points for each electrode
        if self.n > 1:
            points = self.probe.get_random_points_inside(self.n)
            for i, p in enumerate(points):
                # fill in with contact average
                M[i, ] = loop_over_points(p)
            self.recorded_points = points
        else:
            for i, (x, y, z) in enumerate(zip(self.x, self.y, self.z)):
                M[i, ] = self.lfp_method(self.cell,
                                         x=x,
                                         y=y,
                                         z=z,
                                         r_limit=self.cell.d / 2,
                                         sigma=self.sigma,
                                         **kwargs)
            self.recorded_points = np.array([self.x, self.y, self.z]).T

        return M


class OneSphereVolumeConductor(LinearModel):
    """
    Computes extracellular potentials within and outside a spherical volume-
    conductor model that assumes homogeneous, isotropic, linear (frequency
    independent) conductivity in and outside the sphere with a radius R. The
    conductivity in and outside the sphere must be greater than 0, and the
    current source(s) must be located within the radius R.

    The implementation is based on the description of electric potentials of
    point charge in an dielectric sphere embedded in dielectric media [1]_,
    which is mathematically equivalent to a current source in conductive media.

    Parameters
    ----------
    cell: object or None
        CellGeometry instance or similar. If cell==None
    r: ndarray, dtype=float
        shape(3, n_points) observation points in space in spherical coordinates
        (radius, theta, phi) relative to the center of the sphere.
    R: float
        sphere radius [µm]
    sigma_i: float
        electric conductivity for radius r <= R [S/m]
    sigma_o: float
        electric conductivity for radius r > R [S/m]

    References
    ----------
    .. [1] Shaozhong Deng (2008), Journal of Electrostatics 66:549-560.
        DOI: 10.1016/j.elstat.2008.06.003

    Examples
    --------
    Compute the potential for a single monopole along the x-axis:

    >>> # import modules
    >>> from lfpy_forward_models import OneSphereVolumeConductor
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> # observation points in spherical coordinates (flattened)
    >>> X, Y = np.mgrid[-15000:15100:1000., -15000:15100:1000.]
    >>> r = np.array([np.sqrt(X**2 + Y**2).flatten(),
    >>>               np.arctan2(Y, X).flatten(),
    >>>               np.zeros(X.size)])
    >>> # set up class object and compute electric potential in all locations
    >>> sphere = OneSphereVolumeConductor(r, R=10000.,
    >>>                                   sigma_i=0.3, sigma_o=0.03)
    >>> Phi = sphere.calc_potential(rs=8000, current=1.).reshape(X.shape)
    >>> # plot
    >>> fig, ax = plt.subplots(1,1)
    >>> im=ax.contourf(X, Y, Phi,
    >>>                levels=np.linspace(Phi.min(),
    >>>                np.median(Phi[np.isfinite(Phi)]) * 4, 30))
    >>> circle = plt.Circle(xy=(0,0), radius=sphere.R, fc='none', ec='k')
    >>> ax.add_patch(circle)
    >>> fig.colorbar(im, ax=ax)
    >>> plt.show()
    """

    def __init__(self,
                 cell,
                 r,
                 R=10000.,
                 sigma_i=0.3,
                 sigma_o=0.03):
        """initialize class OneSphereVolumeConductor"""
        super().__init__(cell=cell)
        # check inputs
        try:
            assert(r.shape[0] == 3)
            assert(r.ndim == 2)
        except AssertionError:
            raise AssertionError('r must be a shape (3, n_points) ndarray')
        try:
            assert((isinstance(R, float)) or (isinstance(R, int)))
        except AssertionError:
            raise AssertionError('sphere radius R must be a float value')
        try:
            assert((sigma_i > 0) & (sigma_o > 0))
        except AssertionError:
            raise AssertionError(
                'sigma_i and sigma_o must both be positive values')

        self.r = r
        self.R = R
        self.sigma_i = sigma_i
        self.sigma_o = sigma_o

    def calc_potential(self, rs, current, min_distance=1., n_max=1000):
        """
        Return the electric potential at observation points for source current
        as function of time.

        Parameters
        ----------
        rs: float
            monopole source location along the horizontal x-axis [µm]
        current: float or ndarray, dtype float
            float or shape (n_tsteps, ) array containing source current [nA]
        min_distance: None or float
            minimum distance between source location and observation point [µm]
            (in order to avoid singularities)
        n_max: int
            Number of elements in polynomial expansion to sum over (see [1]_).

        References
        ----------
        .. [1] Shaozhong Deng (2008), Journal of Electrostatics 66:549-560.
            DOI: 10.1016/j.elstat.2008.06.003

        Returns
        -------
        Phi: ndarray
            shape (n-points, ) ndarray of floats if I is float like. If I is
            an 1D ndarray, and shape (n-points, I.size) ndarray is returned.
            Unit [mV].
        """
        try:
            assert(type(rs) in [int, float, np.float64])
            assert(abs(rs) < self.R)
        except AssertionError:
            raise AssertionError(
                'source location rs must be a float value and |rs| '
                'must be less than sphere radius R')
        try:
            assert((min_distance is None) or (
                type(min_distance) in [float, int, np.float64]))
        except AssertionError:
            raise AssertionError('min_distance must be None or a float')

        r = self.r[0]
        theta = self.r[1]

        # add harmonical contributions due to inhomogeneous media
        inds_i = r <= self.R
        inds_o = r > self.R

        # observation points r <= R
        phi_i = np.zeros(r.size)
        for j, (theta_i, r_i) in enumerate(zip(theta[inds_i], r[inds_i])):
            coeffs_i = np.zeros(n_max)
            for n in range(n_max):
                coeffs_i[n] = ((self.sigma_i - self.sigma_o) * (n + 1)) / (
                    self.sigma_i * n + self.sigma_o * (n + 1)) * (
                    (r_i * rs) / self.R**2)**n
            poly_i = np.polynomial.legendre.Legendre(coeffs_i)
            phi_i[np.where(inds_i)[0][j]] = poly_i(np.cos(theta_i))
        phi_i[inds_i] *= 1. / self.R

        # observation points r > R
        phi_o = np.zeros(r.size)
        for j, (theta_o, r_o) in enumerate(zip(theta[inds_o], r[inds_o])):
            coeffs_o = np.zeros(n_max)
            for n in range(n_max):
                coeffs_o[n] = (self.sigma_i * (2 * n + 1)) / \
                    (self.sigma_i * n + self.sigma_o * (n + 1)) * (rs / r_o)**n
            poly_o = np.polynomial.legendre.Legendre(coeffs_o)
            phi_o[np.where(inds_o)[0][j]] = poly_o(np.cos(theta_o))
        phi_o[inds_o] *= 1. / r[inds_o]

        # potential in homogeneous media
        if min_distance is None:
            phi_i[inds_i] += 1. / \
                np.sqrt(r[r <= self.R]**2 + rs**2 -
                        2 * r[inds_i] * rs * np.cos(theta[inds_i]))
        else:
            denom = np.sqrt(
                r[inds_i]**2 +
                rs**2 -
                2 *
                r[inds_i] *
                rs *
                np.cos(
                    theta[inds_i]))
            denom[denom < min_distance] = min_distance
            phi_i[inds_i] += 1. / denom

        if isinstance(current, np.ndarray):
            try:
                assert(np.all(np.isfinite(current)))
                assert(np.all(np.isreal(current)))
                assert(current.ndim == 1)
            except AssertionError:
                raise AssertionError('input argument current must be float or '
                                     '1D ndarray with float values')

            return np.dot((phi_i + phi_o).reshape((1, -1)).T,
                          current.reshape((1, -1))
                          ) / (4. * np.pi * self.sigma_i)
        else:
            try:
                assert(np.isfinite(current)) and (np.shape(current) == ())
            except AssertionError:
                raise AssertionError('input argument I must be float or 1D '
                                     'ndarray with float values')
            return current / (4. * np.pi * self.sigma_i) * (phi_i + phi_o)

    def get_response_matrix(self, n_max=1000):
        """
        Compute linear mapping between transmembrane currents of CellGeometry
        like object instance and extracellular potential in and outside of
        sphere.

        Parameters
        ----------
        n_max: int
            Number of elements in polynomial expansion to sum over
            (see [1]_).

        References
        ----------
        .. [1] Shaozhong Deng (2008), Journal of Electrostatics 66:549-560.
            DOI: 10.1016/j.elstat.2008.06.003


        Examples
        --------
        Compute extracellular potential in one-sphere volume conductor model
        from LFPy.Cell object:

        >>> # import modules
        >>> import LFPy
        >>> from lfpy_forward_models import CellGeometry, \
        >>>     OneSphereVolumeConductor
        >>> import os
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib.collections import PolyCollection
        >>> # create cell
        >>> cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0], 'test',
        >>>                                          'ball_and_sticks.hoc'),
        >>>                  tstop=10.)
        >>> cell.set_pos(z=9800.)
        >>> # stimulus
        >>> syn = LFPy.Synapse(cell, idx=cell.totnsegs-1, syntype='Exp2Syn',
        >>>                    weight=0.01)
        >>> syn.set_spike_times(np.array([1.]))
        >>> # simulate
        >>> cell.simulate(rec_imem=True)
        >>> # observation points in spherical coordinates (flattened)
        >>> X, Z = np.mgrid[-500:501:10., 9500:10501:10.]
        >>> Y = np.zeros(X.shape)
        >>> r = np.array([np.sqrt(X**2 + Z**2).flatten(),
        >>>               np.arccos(Z / np.sqrt(X**2 + Z**2)).flatten(),
        >>>               np.arctan2(Y, X).flatten()])
        >>> # instantiate CellGeometry class with cell's geometry
        >>> cell_geometry = CellGeometry(x=np.c_[cell.xstart, cell.xend],
        >>>                              y=np.c_[cell.ystart, cell.yend],
        >>>                              z=np.c_[cell.zstart, cell.zend],
        >>>                              d=cell.diam)
        >>> # set up class object and compute mapping between segment currents
        >>> # and electric potential in space
        >>> sphere = OneSphereVolumeConductor(cell_geometry, r=r, R=10000.,
        >>>                                   sigma_i=0.3, sigma_o=0.03)
        >>> M = sphere.get_response_matrix(n_max=1000)
        >>> # pick out some time index for the potential and compute potential
        >>> ind = cell.tvec==2.
        >>> V_ex = (M @ cell.imem)[:, ind].reshape(X.shape)
        >>> # plot potential
        >>> fig, ax = plt.subplots(1,1)
        >>> zips = []
        >>> for x, z in cell.get_idx_polygons(projection=('x', 'z')):
        >>>     zips.append(list(zip(x, z)))
        >>> polycol = PolyCollection(zips,
        >>>                          edgecolors='none',
        >>>                          facecolors='gray')
        >>> vrange = 1E-3 # limits for color contour plot
        >>> im=ax.contour(X, Z, V_ex,
        >>>              levels=np.linspace(-vrange, vrange, 41))
        >>> circle = plt.Circle(xy=(0,0), radius=sphere.R, fc='none', ec='k')
        >>> ax.add_collection(polycol)
        >>> ax.add_patch(circle)
        >>> ax.axis(ax.axis('equal'))
        >>> ax.set_xlim(X.min(), X.max())
        >>> ax.set_ylim(Z.min(), Z.max())
        >>> fig.colorbar(im, ax=ax)
        >>> plt.show()

        Returns
        -------
        ndarray
            Shape (n_points, n_compartments) mapping between individual
            segments and extracellular potential in extracellular locations

        Notes
        -----
        Each segment is treated as a point source in space. The minimum
        source to measurement site distance will be set to the diameter of
        each segment

        """
        if self.cell is None:
            # perhaps this can be solved with a decorator
            raise Exception('OneSphereVolumeConductor was instantiated with '
                            'cell=None: get_response_matrix() can not be used!'
                            )

        # midpoint position of compartments in spherical coordinates
        radius = np.sqrt(self.cell.x.mean(axis=-1)**2
                         + self.cell.y.mean(axis=-1)**2
                         + self.cell.z.mean(axis=-1)**2)
        theta = np.arccos(self.cell.z.mean(axis=-1) / radius)
        phi = np.arctan2(self.cell.y.mean(axis=-1), self.cell.x.mean(axis=-1))
        diam = self.cell.d

        # since the sources must be located on the x-axis, we keep a copy
        # of the unrotated coordinate system for the contact points:
        r_orig = np.copy(self.r)

        # unit current amplitude
        current = 1.

        # initialize mapping array
        M = np.zeros((self.r.shape[1], radius.size))

        # compute the mapping for each compartment
        for i, (radius_i, theta_i, _, diam_i) in enumerate(
                zip(radius, theta, phi, diam)):
            self.r = np.array([r_orig[0],  # radius unchanged
                               r_orig[1] - theta_i,
                               # rotate relative to source location
                               r_orig[2]])  # phi unchanged
            M[:, i] = self.calc_potential(
                radius_i, current=current, min_distance=diam_i, n_max=n_max)

        # reset measurement locations
        self.r = r_orig

        # return mapping between segment currents and contrib in each
        # measurement location
        return M
