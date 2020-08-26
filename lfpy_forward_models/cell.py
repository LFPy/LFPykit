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
        - each segment is piecewise linear between their start and endpoints
        - each segment has a constant diameter
        - the transmembrane current density is constant along the segment axis


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
        shape (n_seg) array of compartment diameters in units of [um]

    For compatibility with LFPy v2.x, the following class attributes
    will be calculated and set:


    Attributes
    ----------
    totnsegs: int
        number of compartments
    xstart, ystart, zstart: ndarray
        arrays of length totnsegs with start (x,y,z) coordinate of segments
        in units of um
    xmid, ymid, zmid: ndarray
        midpoint coordinates of segments
    xend, yend, zend : ndarray
        endpoint coordinateso of segments
    d: ndarray
        array of length totnsegs with segment diameters in units of um
    length: ndarray
        lenght of each segment in units of um
    area : ndarray
        array of segment surface areas in units of um^2
    '''
    def __init__(self, x, y, z, d):
        '''
        Base class representing the geometry of multicompartment neuron
        models.

        Assumptions
            - each segment is piecewise linear between their
              start and endpoints
            - each segment has a constant diameter
            - the transmembrane current density is constant along the
              segment axis


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
            shape (n_seg) array of compartment diameters in units of [um]

        For compatibility with LFPy v2.x, the following class attributes
        will be calculated and set:


        Attributes
        ----------
        totnsegs: int
            number of compartments
        xstart, ystart, zstart: ndarray
            arrays of length totnsegs with start (x,y,z) coordinate of segments
            in units of um
        xmid, ymid, zmid: ndarray
            midpoint coordinates of segments
        xend, yend, zend : ndarray
            endpoint coordinateso of segments
        d: ndarray
            array of length totnsegs with segment diameters in units of um
        length: ndarray
            lenght of each segment in units of um
        area : ndarray
            array of segment surface areas in units of um^2
        '''
        # check input
        try:
            assert(np.all([type(x) is np.ndarray,
                           type(y) is np.ndarray,
                           type(z) is np.ndarray,
                           type(d) is np.ndarray]))
        except AssertionError:
            raise AssertionError('x, y, z and d must be of type numpy.ndarray')
        try:
            assert(x.ndim == y.ndim == z.ndim == 2)
        except AssertionError:
            raise AssertionError('x, y and z must be of shape (n_seg x 2)')
        try:
            assert(x.shape == y.shape == z.shape)
        except AssertionError:
            raise AssertionError('x, y and z must all be the same shape')
        try:
            assert(x.shape[1] == 2)
        except AssertionError:
            raise AssertionError('the last axis of x, y and z must be of length 2')
        try:
            assert(d.ndim == 1 and d.size == x.shape[0])
        except AssertionError:
            raise AssertionError('d must be 1-dimensional with size == n_seg')

        # set attributes
        self.x = x
        self.y = y
        self.z = z
        self.d = d

        # derived attributes
        self.totnsegs = self.d.size
        # self.xstart = self.x[:, 0]
        # self.xend = self.x[:, -1]
        # self.xmid = self.x.mean(axis=-1)

        # self.ystart = self.y[:, 0]
        # self.yend = self.y[:, -1]
        # self.ymid = self.y.mean(axis=-1)

        # self.zstart = self.z[:, 0]
        # self.zend = self.z[:, -1]
        # self.zmid = self.z.mean(axis=-1)

        self.length = np.sqrt(np.diff(x, axis=-1)**2 +
                              np.diff(y, axis=-1)**2 +
                              np.diff(z, axis=-1)**2)
        self.area = self.length * np.pi * self.d
