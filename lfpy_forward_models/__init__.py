#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Initialization of lfpy_forward_models

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

:LFPy classes to be implemented:
  * RecMEAElectrode:
        Class for simulations of in vitro (slice) extracellular
        potentials
  * OneSphereVolumeConductor:
        For computing extracellular potentials within
        and outside a homogeneous sphere
  * FourSphereVolumeConductor:
        For computing extracellular potentials in
        4-sphere model (brain, CSF, skull, scalp)
  * InfiniteVolumeConductor:
        To compute extracellular potentials with current
        dipoles in infinite volume conductor
  * MEG:
        Class for computing magnetic field from current dipole moments

:Modules:
  * cell
  * models
  * lfpcalc
"""

from .version import version as __version__

from .cell import CellGeometry
from .models import LinearModel, CurrentDipoleMoment, PointSourcePotential, \
    LineSourcePotential, RecExtElectrode
