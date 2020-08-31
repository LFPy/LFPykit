lfpy_forward_models
===================

This Python module contain freestanding implementations of electrostatic
forward models presently incorporated in LFPy
(https://github.com/LFPy/LFPy, https://LFPy.readthedocs.io).

The aim is to provide electrostatic models in a manner that facilitates
forward-model predictions of extracellular potentials and related measures from
multicompartment neuron models, but without explicit dependencies on neural
simulation software such as NEURON
(https://neuron.yale.edu, https://github.com/neuronsimulator/nrn)
or Arbor (https://arbor.readthedocs.io, https://github.com/arbor-sim/arbor).

Build Status
------------

[![Build Status](https://travis-ci.org/LFPy/lfpy_forward_models.svg?branch=master)](https://travis-ci.org/LFPy/lfpy_forward_models)
[![Coverage Status](https://coveralls.io/repos/github/LFPy/lfpy_forward_models/badge.svg?branch=master)](https://coveralls.io/github/LFPy/lfpy_forward_models?branch=master)
[![Documentation Status](https://readthedocs.org/projects/lfpy-forward-models/badge/?version=latest)](https://lfpy-forward-models.readthedocs.io/en/latest/?badge=latest)
![Lintly flake8 checks](https://github.com/LFPy/lfpy_forward_models/workflows/Lintly%20flake8%20checks/badge.svg)
![Python application](https://github.com/LFPy/lfpy_forward_models/workflows/Python%20application/badge.svg)
![Upload Python Package](https://github.com/LFPy/lfpy_forward_models/workflows/Upload%20Python%20Package/badge.svg)

Features
--------

`lfpy_forward_models` presently incorporates different electrostatic forward models for extracellular potentials
and magnetic signals that has been derived using volume conductor theory.
In volume-conductor theory the extracellular potentials can be calculated from a distance-weighted sum of contributions from transmembrane currents of neurons.
Given the same transmembrane currents, the contributions to the magnetic field recorded both inside and outside the brain can also be computed.

The module presently incorporates different classes.
To represent the geometry of a multicompartment neuron model we have:

* `CellGeometry`:
  Base class representing a multicompartment neuron geometry in terms
  of segment x-, y-, z-coordinates and diameter.

Different classes built to map transmembrane currents of `CellGeometry` like instances
to different measurement modalities:

* `LinearModel`:
  ase class representing a generic forward model
  for subclassing
* `CurrentDipoleMoment`:
  Class for predicting current dipole moments
* `PointSourcePotential`:
  Class for predicting extracellular potentials
  assuming point sources and point contacts
* `LineSourcePotential`:
  Class for predicting extracellular potentials assuming
  line sourcers and point contacts
* `RecExtElectrode`:
  Class for simulations of extracellular potentials
* `RecMEAElectrode`:
  Class for simulations of in vitro (slice) extracellular
  potentials
* `OneSphereVolumeConductor`:
  For computing extracellular potentials within
  sand outside a homogeneous sphere

Different classes built to map current dipole moments (e.g., computed using `CurrentDipoleMoment`)
to extracellular measurements:

* `eegmegcalc.FourSphereVolumeConductor`:
  For computing extracellular potentials in
  4-sphere head model (brain, CSF, skull, scalp)
  from current dipole moment
* `eegmegcalc.InfiniteVolumeConductor`:
  To compute extracellular potentials in infinite volume conductor
  from current dipole moment
* `eegmegcalc.MEG`:
  Class for computing magnetic field from current dipole moments


Usage
-----

A basic usage example using a mock 3-segment stick like neuron,
treating each segment as a point source in a linear, isotropic and homogeneous volume conductor,
computing the extracellular potential in ten different locations
alongside the cell geometry:

    >>> # imports
    >>> import numpy as np
    >>> from lfpy_forward_models import CellGeometry, PointSourcePotential
    >>> n_seg = 3
    >>> # instantiate class `CellGeometry`:
    >>> cell = CellGeometry(x=np.array([[0.] * 2] * n_seg),  # (µm)
                            y=np.array([[0.] * 2] * n_seg),  # (µm)
                            z=np.array([[10. * x, 10. * (x + 1)]
                                        for x in range(n_seg)]),  # (µm)
                            d=np.array([1.] * n_seg))  # (µm)
    >>> # instantiate class `PointSourcePotential`:
    >>> psp = PointSourcePotential(cell,
                                   x=np.ones(10) * 10,
                                   y=np.zeros(10),
                                   z=np.arange(10) * 10,
                                   sigma=0.3)
    >>> # get linear response matrix mapping currents to measurements
    >>> M = psp.get_response_matrix()
    >>> # transmembrane currents (nA):
    >>> imem = np.array([[-1., 1.],
                         [0., 0.],
                         [1., -1.]])
    >>> # compute extracellular potentials (mV)
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


A basic usage example using a mock 3-segment stick like neuron,
treating each segment as a point source,
computing the current dipole moment and computing the potential in ten different
remote locations away from the cell geometry:

    >>> # imports
    >>> import numpy as np
    >>> from lfpy_forward_models import CellGeometry, CurrentDipoleMoment, \
    >>>     eegmegcalc
    >>> n_seg = 3
    >>> # instantiate class `CellGeometry`:
    >>> cell = CellGeometry(x=np.array([[0.] * 2] * n_seg),  # (µm)
                            y=np.array([[0.] * 2] * n_seg),  # (µm)
                            z=np.array([[10. * x, 10. * (x + 1)]
                                        for x in range(n_seg)]),  # (µm)
                            d=np.array([1.] * n_seg))  # (µm)
    >>> # instantiate class `CurrentDipoleMoment`:
    >>> cdp = CurrentDipoleMoment(cell)
    >>> M_I_to_P = cdp.get_response_matrix()
    >>> # instantiate class `eegmegcalc.InfiniteVolumeConductor` and map dipole moment to
    >>> # extracellular potential at measurement sites
    >>> ivc = eegmegcalc.InfiniteVolumeConductor(sigma=0.3)
    >>> # compute linear response matrix between dipole moment and
    >>> # extracellular potential
    >>> M_P_to_V = ivc.get_response_matrix(np.c_[np.ones(10) * 1000,
                                                 np.zeros(10),
                                                 np.arange(10) * 100])
    >>> # transmembrane currents (nA):
    >>> imem = np.array([[-1., 1.],
                        [0., 0.],
                        [1., -1.]])
    >>> # compute extracellular potentials (mV)
    >>> V_ex = M_P_to_V @ M_I_to_P @ imem
    >>> V_ex
    array([[ 0.00000000e+00,  0.00000000e+00],
          [ 5.22657054e-07, -5.22657054e-07],
          [ 1.00041193e-06, -1.00041193e-06],
          [ 1.39855769e-06, -1.39855769e-06],
          [ 1.69852477e-06, -1.69852477e-06],
          [ 1.89803345e-06, -1.89803345e-06],
          [ 2.00697409e-06, -2.00697409e-06],
          [ 2.04182029e-06, -2.04182029e-06],
          [ 2.02079888e-06, -2.02079888e-06],
          [ 1.96075587e-06, -1.96075587e-06]])


Documentation
-------------

The online Documentation of `lfpy_forward_models` can be found here:
https://lfpy-forward-models.readthedocs.io/en/latest


dependencies
------------

`lfpy_forward_models` is implemented in Python and is written (and continuously) tested for Python >= v3.7.
The main `lfpy_forward_models` module depends on `numpy`, `scipy` and `MEAutility` (https://github.com/alejoe91/MEAutility, https://meautility.readthedocs.io/en/latest/).

Running all unit tests and example files may in addition require `py.test`, `matplotlib`,
`neuron` (https://www.neuron.yale.edu),
(`arbor` coming) and
`LFPy` (https://github.com/LFPy/LFPy, https://LFPy.readthedocs.io).


Installation
------------

First, make sure that the above dependencies are installed in the current Python environment.

Install the current development version on https://GitHub.com using `git` (https://git-scm.com):

    $ git clone https://github.com/LFPy/lfpy_forward_models.git
    $ cd lfpy_forward_models
    $ python setup.py install  # --user optional

or using `pip`:

    $ pip install .  # --user optional

For active development, link the repository location

    $ python setup.py develop  # --user optional

Installing from the Python Package Index (pypi.org):

    $ pip install lfpy_forward_models  # --user optional
