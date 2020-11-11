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


class CellGeometry(object):
    '''
    Base class representing the geometry of multicompartment neuron
    models.

    Assumptions
        - each compartment is piecewise linear between their start
          and endpoints
        - each compartment has a constant diameter
        - the transmembrane current density is constant along the
          compartment axis

    Parameters
    ----------
    x: ndarray of floats
        shape (n_seg x 2) array of start- and end-point coordinates of
        each compartment along x-axis in units of [um]
    y: ndarray
        shape (n_seg x 2) array of start- and end-point coordinates of
        each compartment along y-axis in units of [um]
    z: ndarray
        shape (n_seg x 2) array of start- and end-point coordinates of
        each compartment along z-axis in units of [um]
    d: ndarray
        shape (n_seg) or shape (n_seg x 2) array of compartment
        diameters in units of [um]. If the 2nd axis is equal to 2,
        conical frusta is assumed.

    Attributes
    ----------
    totnsegs: int
        number of compartments
    x: ndarray of floats
        shape (totnsegs x 2) array of start- and end-point coordinates of
        each compartment along x-axis in units of [um]
    y: ndarray
        shape (totnsegs x 2) array of start- and end-point coordinates of
        each compartment along y-axis in units of [um]
    z: ndarray
        shape (totnsegs x 2) array of start- and end-point coordinates of
        each compartment along z-axis in units of [um]
    d: ndarray
        shape (totnsegs) array of compartment diameters in units of [um]
    length: ndarray
        lenght of each compartment in units of um
    area: ndarray
        array of compartment surface areas in units of um^2
    '''
    def __init__(self, x, y, z, d):
        '''
        Base class representing the geometry of multicompartment neuron
        models.

        Assumptions
            - each compartment is piecewise linear between their
              start and endpoints
            - each compartment has a constant diameter
            - the transmembrane current density is constant along the
              compartment axis

        Parameters
        ----------
        x: ndarray of floats
            shape (n_seg x 2) array of start- and end-point coordinates of
            each compartment along x-axis in units of [um]
        y: ndarray
            shape (n_seg x 2) array of start- and end-point coordinates of
            each compartment along y-axis in units of [um]
        z: ndarray
            shape (n_seg x 2) array of start- and end-point coordinates of
            each compartment along z-axis in units of [um]
        d: ndarray
            shape (n_seg) or shape (n_seg x 2) array of compartment
            diameters in units of [um]. If the 2nd axis is equal to 2,
            conical frusta is assumed.

        Attributes
        ----------
        totnsegs: int
            number of compartments
        x: ndarray of floats
            shape (totnsegs x 2) array of start- and end-point coordinates of
            each compartment along x-axis in units of [um]
        y: ndarray
            shape (totnsegs x 2) array of start- and end-point coordinates of
            each compartment along y-axis in units of [um]
        z: ndarray
            shape (totnsegs x 2) array of start- and end-point coordinates of
            each compartment along z-axis in units of [um]
        d: ndarray
            shape (n_seg) or shape (n_seg x 2) array of compartment
            diameters in units of [um]. If the 2nd axis is equal to 2,
            conical frusta is assumed.
        length: ndarray
            lenght of each compartment in units of um
        area: ndarray
            array of compartment surface areas in units of um^2
        '''
        # check input
        assert np.all([type(x) is np.ndarray,
                       type(y) is np.ndarray,
                       type(z) is np.ndarray,
                       type(d) is np.ndarray]), \
            'x, y, z and d must be of type numpy.ndarray'
        assert x.ndim == y.ndim == z.ndim == 2, \
            'x, y and z must be of shape (n_seg x 2)'
        assert x.shape == y.shape == z.shape, \
            'x, y and z must all be the same shape'
        assert x.shape[1] == 2, \
            'the last axis of x, y and z must be of length 2'
        assert d.shape == x.shape or (d.ndim == 1 and d.size == x.shape[0]), \
            'd must either be 1-dimensional with size == n_seg ' + \
            'or 2-dimensional with d.shape == x.shape'

        # set attributes
        self.x = x
        self.y = y
        self.z = z
        self.d = d

        # derived attributes
        self.totnsegs = self.x.shape[0]

        self._set_length()
        self._set_area()

    def _set_length(self):
        self.length = np.sqrt(np.diff(self.x, axis=-1)**2 +
                              np.diff(self.y, axis=-1)**2 +
                              np.diff(self.z, axis=-1)**2).flatten()

    def _set_area(self):
        if self.d.ndim == 1:
            self.area = self.length * np.pi * self.d
        else:
            # Surface area of conical frusta
            # A = pi*(r1+r2)*sqrt((r1-r2)^2 + h^2)
            self.area = np.pi * self.d.sum(axis=-1) * \
                np.sqrt(np.diff(self.d, axis=-1)**2 + self.length**2)
