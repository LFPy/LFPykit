#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Initialization of LFPykit

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

:Classes:
  * CellGeometry:
        Base class representing a multicompartment neuron geometry
        for subclassing
  * LinearModel:
        Base class representing a generic forward model
        for subclassing
  * CurrentDipoleMoment:
        Class for predicting current dipole moments
  * PointSourcePotential:
        Class for predicting extracellular potentials
        assuming point sources and contacts
  * LineSourcePotential:
        Class for predicting extracellular potentials assuming
        line sourcers and point contacts
  * RecExtElectrode:
        Class for simulations of extracellular potentials
  * RecMEAElectrode:
        Class for simulations of in vitro (slice) extracellular
        potentials
  * OneSphereVolumeConductor:
        For computing extracellular potentials within
        and outside a homogeneous sphere
  * LaminarCurrentSourceDensity:
        For computing the ground truth current source density in cylindrical
        volumes aligned with the z-axis.
  * VolumetricCurrentSourceDensity:
        For computing the ground truth current source density in cubic volumes
        with bin edges defined by x, y, z
  * eegmegcalc.FourSphereVolumeConductor:
        For computing extracellular potentials in
        4-sphere model (brain, CSF, skull, scalp) from current dipole moment
  * eegmegcalc.InfiniteVolumeConductor:
        To compute extracellular potentials with current
        dipole moments in infinite volume conductor
  * eegmegcalc.MEG:
        Class for computing magnetic field from current dipole moments
  * `eegmegcalc.NYHeadModel`:
        Class for computing extracellular potentials in detailed head volume
        conductor model (https://www.parralab.org/nyhead)

:Modules:
  * cellgeometry
  * models
  * eegmegcalc
  * lfpcalc
"""

from .version import version as __version__

from .cellgeometry import CellGeometry
from .models import LinearModel, CurrentDipoleMoment, PointSourcePotential, \
    LineSourcePotential, RecExtElectrode, RecMEAElectrode, \
    OneSphereVolumeConductor, LaminarCurrentSourceDensity, \
    VolumetricCurrentSourceDensity
from . import eegmegcalc
