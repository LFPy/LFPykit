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
          Base class representing a multicompartment neuron geometry

Different models built to map transmembrane currents to measurement:

    * `LinearModel`:
          Base class representing a generic forward model
          for subclassing
    * `CurrentDipoleMoment`:
          Class for predicting current dipole moments
    * `PointSourcePotential`:
          Class for predicting extracellular potentials
          assuming point sources and contacts
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
          and outside a homogeneous sphere

Different models built to map current dipole moments to measurements:

    * `eegmegcalc.FourSphereVolumeConductor`:
          For computing extracellular potentials in
          4-sphere head model (brain, CSF, skull, scalp)
          from current dipole moment
    * `eegmegcalc.InfiniteVolumeConductor`:
          To compute extracellular potentials in infinite volume conductor
          from current dipole moment
    * `eegmegcalc.MEG`:
          Class for computing magnetic field from current dipole moments

    :LFPy classes to be implemented:



Documentation
-------------

The online Documentation of `lfpy_forward_models` can be found here:
https://lfpy-forward-models.readthedocs.io/en/latest


dependencies
------------

`lfpy_forward_models` is implemented in Python and is written (and continuously) tested for Python >= v3.7.
The main `lfpy_forward_models` module depends on `numpy`, `scipy` and `MEAutility` (https://github.com/alejoe91/MEAutility, https://meautility.readthedocs.io/en/latest/).

Running all unit tests and example files may in addition require `py.test`, `matplotlib`, `neuron`, `arbor` and `LFPy`.


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

Installing from the Python package index:

    $ pip install lfpy_forward_models  # --user optional
