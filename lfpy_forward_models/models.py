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
            assert(np.all([isinstance(x, np.ndarray),
                           isinstance(y, np.ndarray),
                           isinstance(z, np.ndarray)]))
        except AssertionError:
            raise AssertionError('x, y and z must be of type numpy.ndarray')
        try:
            assert(x.ndim == y.ndim == z.ndim == 1)
        except AssertionError:
            raise AssertionError('x, y and z must be of shape (n_coords, )')
        try:
            assert(x.shape == y.shape == z.shape)
        except AssertionError:
            raise AssertionError(
                'x, y and z must contain the same number of elements')
        try:
            assert(isinstance(sigma, float) and sigma > 0)
        except AssertionError:
            raise AssertionError(
                'sigma must be a float number greater than zero')

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
            assert(np.all([isinstance(x, np.ndarray),
                           isinstance(y, np.ndarray),
                           isinstance(z, np.ndarray)]))
        except AssertionError:
            raise AssertionError('x, y and z must be of type numpy.ndarray')
        try:
            assert(x.ndim == y.ndim == z.ndim == 1)
        except AssertionError:
            raise AssertionError('x, y and z must be of shape (n_coords, )')
        try:
            assert(x.shape == y.shape == z.shape)
        except AssertionError:
            raise AssertionError(
                'x, y and z must contain the same number of elements')
        try:
            assert(isinstance(sigma, float) and sigma > 0)
        except AssertionError:
            raise AssertionError(
                'sigma must be a float number greater than zero')

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
    >>> from lfpy_forward_models import RecExtElectrode
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
    >>> N = np.empty((16, 3))
    >>> for i in xrange(N.shape[0]): N[i,] = [1, 0, 0] # normal vectors
    >>> electrodeParameters = {         # parameters for RecExtElectrode class
    >>>     'sigma' : 0.3,              # Extracellular potential
    >>>     'x' : np.zeros(16)+25,      # Coordinates of electrode contacts
    >>>     'y' : np.zeros(16),
    >>>     'z' : np.linspace(-500,1000,16),
    >>>     'n' : 20,
    >>>     'r' : 10,
    >>>     'N' : N,
    >>> }
    >>> electrode = RecExtElectrode(cell, **electrodeParameters)
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
    >>> from lfpy_forward_models import RecExtElectrode
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
    >>> for i in xrange(N.shape[0]): N[i,] = [1, 0, 0] #normal vec. of contacts
    >>> electrodeParameters = {         # parameters for RecExtElectrode class
    >>>     'sigma' : 0.3,              # Extracellular potential
    >>>     'x' : np.zeros(16)+25,      # Coordinates of electrode contacts
    >>>     'y' : np.zeros(16),
    >>>     'z' : np.linspace(-500,1000,16),
    >>>     'n' : 20,
    >>>     'r' : 10,
    >>>     'N' : N,
    >>> }
    >>> electrode = LFPy.RecExtElectrode(cell, **electrodeParameters)
    >>> cell.simulate(dotprodcoeffs=[electrode.get_response_matrix()])
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
    >>> from lfpy_forward_models import RecExtElectrode
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
    >>> electrode = RecExtElectrode(cell, probe=probe)
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
            except AssertionError:
                raise AssertionError(
                    "The number of elements in [x, y, z] must be equal")

            if N is not None:
                if not isinstance(N, np.array):
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
            shape (n_contacts, n_seg) ndarray
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


class RecMEAElectrode(RecExtElectrode):
    r"""class RecMEAElectrode

    Electrode class that represents an extracellular in vitro slice recording
    as a Microelectrode Array (MEA). Inherits RecExtElectrode class

    Illustration:
    ::

                  Above neural tissue (Saline) -> sigma_S
        <----------------------------------------------------> z = z_shift + h

                  Neural Tissue -> sigma_T

                       o -> source_pos = [x',y',z']

        <-----------*----------------------------------------> z = z_shift + 0
                     \-> elec_pos = [x,y,z]

                  Below neural tissue (MEA Glass plate) -> sigma_G


    Parameters
    ----------
    cell: object
        GeometryCell instance or similar.
    sigma_T: float
        extracellular conductivity of neural tissue in unit (S/m)
    sigma_S: float
        conductivity of saline bath that the neural slice is
        immersed in [1.5] (S/m)
    sigma_G: float
        conductivity of MEA glass electrode plate. Most commonly
        assumed non-conducting [0.0] (S/m)
    h: float, int
        Thickness in um of neural tissue layer containing current
        the current sources (i.e., in vitro slice or cortex)
    z_shift: float, int
        Height in um of neural tissue layer bottom. If e.g., top of neural
        tissue layer should be z=0, use z_shift=-h. Defaults to z_shift = 0, so
        that the neural tissue layer extends from z=0 to z=h.
    squeeze_cell_factor: float or None
        Factor to squeeze the cell in the z-direction. This is
        needed for large cells that are thicker than the slice, since no part
        of the cell is allowed to be outside the slice. The squeeze is done
        after the neural simulation, and therefore does not affect neuronal
        simulation, only calculation of extracellular potentials.
    probe: MEAutility MEA object or None
        MEAutility probe object
    x, y, z: np.ndarray
        coordinates or arrays of coordinates in units of (um).
        Must be same length
    N: None or list of lists
        Normal vectors [x, y, z] of each circular electrode contact surface,
        default None
    r: float
        radius of each contact surface, default None
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
    >>> from lfpy_forward_models import CellGeometry, RecMEAElectrode
    >>> # cell geometry with four segments [um]
    >>> cell = CellGeometry(
    >>>     x=np.array([[0, 10], [10, 20], [20, 30], [30, 40]]),
    >>>     y=np.array([[0, 0], [0, 0], [0, 0], [0, 0]]),
    >>>     z=np.array([[0, 0], [0, 0], [0, 0], [0, 0]]) * 10,
    >>>     d=np.array([1, 1, 1, 1]))
    >>> # transmembrane currents, three time steps [nA]
    >>> I_m = np.array([[0.25, -1., 1.],
    >>>                 [-1., 1., -0.25],
    >>>                 [1., -0.25, -1.],
    >>>                 [-0.25, 0.25, 0.25]])
    >>> # electrode locations [um]
    >>> r = np.stack([np.arange(10)*4 + 2, np.zeros(10), np.zeros(10)])
    >>> # instantiate electrode, get linear response matrix
    >>> el = RecMEAElectrode(cell=cell,
    >>>                      sigma_T=0.3, sigma_S=1.5, sigma_G=0.0,
    >>>                      x=r[0, ], y=r[1, ], z=r[2, ],
    >>>                      method='pointsource')
    >>> M = el.get_response_matrix()
    >>> # compute extracellular potential
    >>> M @ I_m
    array([[ 0.02244775, -0.13777683,  0.14758927],
           [ 0.09703145, -0.47397696,  0.49243115],
           [-0.04951479, -0.00353662,  0.04951479],
           [-0.47386676,  0.46582868, -0.11559584],
           [-0.09865022,  0.12488441, -0.07138681],
           [ 0.09865022,  0.01057345, -0.15437628],
           [ 0.47386676, -0.09492693, -0.50257376],
           [ 0.04951479,  0.0141465 , -0.06719792],
           [-0.09703145,  0.12752004,  0.08499705],
           [-0.02244775,  0.04099623,  0.01371172]])

    See also <LFPy>/examples/example_MEA.py

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import LFPy
    >>> from lfpy_forward_models import CellGeometry, RecMEAElectrode
    >>>
    >>> cellParameters = {
    >>>     'morphology' : 'examples/morphologies/L5_Mainen96_LFPy.hoc',
    >>>     'v_init' : -65,                          # initial voltage
    >>>     'cm' : 1.0,                             # membrane capacitance
    >>>     'Ra' : 150,                             # axial resistivity
    >>>     'passive' : True,                        # insert passive channels
    >>>     'passive_parameters' : {"g_pas":1./3E4,
    >>>                             "e_pas":-65}, # passive params
    >>>     'dt' : 2**-4,                           # simulation time res
    >>>     'tstart' : 0.,                        # start t of simulation
    >>>     'tstop' : 50.,                        # end t of simulation
    >>> }
    >>> lfpy_cell = LFPy.Cell(**cellParameters)
    >>> lfpy_cell.set_rotation(x=np.pi/2, z=np.pi/2)
    >>> lfpy_cell.set_pos(z=100)
    >>> synapseParameters = {
    >>>     'idx' : lfpy_cell.get_closest_idx(x=800, y=0, z=100), # segment
    >>>     'e' : 0,                                # reversal potential
    >>>     'syntype' : 'ExpSyn',                   # synapse type
    >>>     'tau' : 2,                              # syn. time constant
    >>>     'weight' : 0.01,                       # syn. weight
    >>>     'record_current' : True                 # syn. current record
    >>> }
    >>> synapse = LFPy.Synapse(lfpy_cell, **synapseParameters)
    >>> synapse.set_spike_times(np.array([10., 15., 20., 25.]))
    >>>
    >>> MEA_electrode_parameters = {
    >>>     'sigma_T' : 0.3,      # extracellular conductivity
    >>>     'sigma_G' : 0.0,      # MEA glass electrode plate conductivity
    >>>     'sigma_S' : 1.5,      # Saline bath conductivity
    >>>     'x' : np.linspace(0, 1200, 16),  # 1d vector of positions
    >>>     'y' : np.zeros(16),
    >>>     'z' : np.zeros(16),
    >>>     "method": "pointsource",
    >>>     "h": 300,
    >>>     "squeeze_cell_factor": 0.3,
    >>> }
    >>> lfpy_cell.simulate(rec_imem=True)
    >>>
    >>> cell = CellGeometry(
    >>>     x=np.c_[lfpy_cell.xstart, lfpy_cell.xend],
    >>>     y=np.c_[lfpy_cell.ystart, lfpy_cell.yend],
    >>>     z=np.c_[lfpy_cell.zstart, lfpy_cell.zend],
    >>>     d=lfpy_cell.diam)
    >>> MEA = RecMEAElectrode(cell, **MEA_electrode_parameters)
    >>> V_ext = MEA.get_response_matrix() @ lfpy_cell.imem
    >>>
    >>> plt.matshow(V_ext)
    >>> plt.colorbar()
    >>> plt.axis('tight')
    >>> plt.show()
    """

    def __init__(self, cell, sigma_T=0.3, sigma_S=1.5, sigma_G=0.0,
                 h=300., z_shift=0., steps=20, probe=None,
                 x=np.array([0]), y=np.array([0]), z=np.array([0]),
                 N=None, r=None, n=None,
                 method='linesource',
                 verbose=False,
                 seedvalue=None, squeeze_cell_factor=None, **kwargs):

        super().__init__(cell=cell,
                         x=x, y=y, z=z,
                         probe=probe,
                         N=N, r=r, n=n,
                         method=method,
                         verbose=verbose,
                         seedvalue=seedvalue, **kwargs)

        self.sigma_G = sigma_G
        self.sigma_T = sigma_T
        self.sigma_S = sigma_S
        self.sigma = None
        self.h = h
        self.z_shift = z_shift
        self.steps = steps
        self.squeeze_cell_factor = squeeze_cell_factor
        self.moi_param_kwargs = {"h": self.h,
                                 "steps": self.steps,
                                 "sigma_G": self.sigma_G,
                                 "sigma_T": self.sigma_T,
                                 "sigma_S": self.sigma_S,
                                 }

        if method == 'pointsource':
            self.lfp_method = lfpcalc.calc_lfp_pointsource_moi
        elif method == "linesource":
            if (np.abs(z - self.z_shift) > 1e-9).any():
                raise NotImplementedError("The method 'linesource' is only "
                                          "supported for electrodes at the "
                                          "z=0 plane. Use z=0 or method "
                                          "'pointsource'.")
            if np.abs(self.sigma_G) > 1e-9:
                raise NotImplementedError("The method 'linesource' is only "
                                          "supported for sigma_G=0. Use "
                                          "sigma_G=0 or method "
                                          "'pointsource'.")
            self.lfp_method = lfpcalc.calc_lfp_linesource_moi
        elif method == "root_as_point":
            if (np.abs(z - self.z_shift) > 1e-9).any():
                raise NotImplementedError("The method 'root_as_point' is only "
                                          "supported for electrodes at the "
                                          "z=0 plane. Use z=0 or method "
                                          "'pointsource'.")
            if np.abs(self.sigma_G) > 1e-9:
                raise NotImplementedError("The method 'root_as_point' is only "
                                          "supported for sigma_G=0. Use "
                                          "sigma_G=0 or method "
                                          "'pointsource'.")
            self.lfp_method = lfpcalc.calc_lfp_root_as_point_moi
        else:
            raise ValueError("LFP method not recognized. "
                             "Should be 'root_as_point', 'linesource' "
                             "or 'pointsource'")

    def _squeeze_cell_in_depth_direction(self):
        """Will squeeze self.cell centered around the root segment by a scaling
        factor, so that it fits inside the slice. If scaling factor is not big
        enough, a RuntimeError is raised."""

        self.distort_cell_geometry()

        if (self.cell.z.max() > self.h + self.z_shift or
                self.cell.z.min() < self.z_shift):
            bad_comps, reason = self._return_comp_outside_slice()
            msg = ("Compartments {} of cell ({}) has cell.{} slice. "
                   "Increase squeeze_cell_factor, move or rotate cell."
                   ).format(bad_comps, self.cell, reason)

            raise RuntimeError(msg)

    def _return_comp_outside_slice(self):
        """
        Assuming part of the cell is outside the valid region,
        i.e, not in the slice (self.z_shift < z < self.z_shift + self.h)
        this function check what array (cell.z[:, 0] or cell.z[:, -1]) that is
        outside, and if it is above or below the valid region.

        Raises: RuntimeError
            If no compartment is outside valid region.

        Returns: array, str
            Numpy array with the compartments that are outside the slice,
            and a string with additional information on the problem.
        """
        zstart_above = np.where(self.cell.z[:, 0] > self.z_shift + self.h)[0]
        zend_above = np.where(self.cell.z[:, -1] > self.z_shift + self.h)[0]
        zend_below = np.where(self.cell.z[:, -1] < self.z_shift)[0]
        zstart_below = np.where(self.cell.z[:, 0] < self.z_shift)[0]

        if len(zstart_above) > 0:
            return zstart_above, "zstart above"
        if len(zstart_below) > 0:
            return zstart_below, "zstart below"
        if len(zend_above) > 0:
            return zend_above, "zend above"
        if len(zend_below) > 0:
            return zend_below, "zend below"
        raise RuntimeError("This function should only be called if cell"
                           "extends outside slice")

    def _test_cell_extent(self):
        """
        Test if the cell is confined within the slice.
        If class argument "squeeze_cell" is True, cell is squeezed to to
        fit inside slice.
        """
        if self.cell is None:
            raise RuntimeError("Does not have cell instance.")

        if (self.cell.z.max() > self.z_shift + self.h or
                self.cell.z.min() < self.z_shift):

            if self.verbose:
                print("Cell extends outside slice.")

            if self.squeeze_cell_factor is not None:
                if not self.z_shift < self.cell.z[0, ].mean() < \
                        (self.z_shift + self.h):
                    raise RuntimeError("Soma position is not in slice.")
                self._squeeze_cell_in_depth_direction()
            else:
                bad_comps, reason = self._return_comp_outside_slice()
                msg = ("Compartments {} of cell ({}) has cell.{} slice "
                       "and argument squeeze_cell_factor is None."
                       ).format(bad_comps, self.cell, reason)
                raise RuntimeError(msg)
        else:
            if self.verbose:
                print("Cell position is good.")
            if self.squeeze_cell_factor is not None:
                if self.verbose:
                    print("Squeezing cell anyway.")
                self._squeeze_cell_in_depth_direction()

    def distort_cell_geometry(self, axis='z', nu=0.0):
        """
        Distorts cellular morphology with a relative squeeze_cell_factor along
        a chosen axis preserving Poisson's ratio. A ratio nu=0.5 assumes
        uncompressible and isotropic media that embeds the cell. A ratio nu=0
        will only affect geometry along the chosen axis. A ratio nu=-1 will
        isometrically scale the neuron geometry along each axis.
        This method does not affect the underlying cable properties of the
        cell, only predictions of extracellular measurements (by affecting the
        relative locations of sources representing the compartments).

        Parameters
        ----------
        axis : str
            which axis to apply compression/stretching. Default is "z".
        nu : float
            Poisson's ratio. Ratio between axial and transversal
            compression/stretching. Default is 0.
        """
        try:
            assert(abs(self.squeeze_cell_factor) < 1.)
        except AssertionError:
            raise AssertionError('abs(squeeze_cell_factor) >= 1, '
                                 + ' squeeze_cell_factor must be in <-1, 1>')
        try:
            assert(axis in ['x', 'y', 'z'])
        except AssertionError:
            raise AssertionError('axis={} not "x", "y" or "z"'.format(axis))

        for pos, dir_ in zip([self.cell.x[0, ].mean(),
                              self.cell.y[0, ].mean(),
                              self.cell.z[0, ].mean()],
                             'xyz'):
            geometry = getattr(self.cell, dir_)
            if dir_ == axis:
                geometry -= pos
                geometry *= (1. - self.squeeze_cell_factor)
                geometry += pos
            else:
                geometry -= pos
                geometry *= (1. + self.squeeze_cell_factor * nu)
                geometry += pos
            setattr(self.cell, dir_, geometry)

        # recompute length of each segment
        self.cell.length = np.sqrt(np.diff(self.cell.x, axis=-1)**2
                                   + np.diff(self.cell.y, axis=-1)**2
                                   + np.diff(self.cell.z, axis=-1)**2)
        # recompute segment areas
        self.cell.area = self.cell.length * np.pi * self.cell.d

    def get_response_matrix(self):
        '''
        Get linear response matrix

        Returns
        -------
        response_matrix: ndarray
            shape (n_contacts, n_seg) ndarray
        '''

        self._test_cell_extent()

        # Temporarily shift coordinate system so middle layer extends
        # from z=0 to z=h
        self.z = self.z - self.z_shift
        self.cell.z = self.cell.z - self.z_shift

        if self.n is not None and self.N is not None and self.r is not None:
            if self.n <= 1:
                raise ValueError("n = %i must be larger that 1" % self.n)
            else:
                pass

            M = self._lfp_el_pos_calc_dist(**self.moi_param_kwargs)

            if self.verbose:
                print('calculations finished, %s, %s' % (str(self),
                                                         str(self.cell)))
        else:
            M = self._loop_over_contacts(**self.moi_param_kwargs)
            if self.verbose:
                print('calculations finished, %s, %s' % (str(self),
                                                         str(self.cell)))

        # Shift coordinate system back so middle layer extends
        # from z=z_shift to z=z_shift + h
        self.z = self.z + self.z_shift
        self.cell.z = self.cell.z + self.z_shift

        return M
