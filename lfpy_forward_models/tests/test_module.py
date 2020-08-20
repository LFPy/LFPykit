#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Copyright (C) 2020 Computational Neuroscience Group, NMBU.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

"""

import unittest
import numpy as np
import lfpy_forward_models as lfp


def get_cell(n_seg=4):
    cell = lfp.CellGeometry(x=np.array([[0.]*2]*n_seg),
                            y=np.array([[0.]*2]*n_seg),
                            z=np.array([[1.*x, 1.*(x+1)]
                                        for x in range(n_seg)]),
                            diam=np.array([1.]*n_seg))
    return cell


class TestSuite(unittest.TestCase):
    """
    test methods and modules
    """
    def test_TestSuite_00(self):
        '''test TestSuite'''
        self.assertTrue(True)

    def test_LinearModel_00(self):
        '''test LinearModel'''
        cell = get_cell(n_seg=4)
        lm = lfp.LinearModel(cell)
        M = lm.get_response_matrix()

        imem = np.arange(24).reshape((4, -1))

        self.assertTrue(np.all(imem == np.dot(M, imem)))

    def test_CurrentDipoleMoment_00(self):
        '''test CurrentDipoleMoment'''
        cell = get_cell(n_seg=3)
        cdm = lfp.CurrentDipoleMoment(cell)
        M = cdm.get_response_matrix()

        imem = np.array([[-1., 1.],
                         [0., 0.],
                         [1., -1.]])

        P = np.dot(M, imem)

        P_gt = np.array([[0., 0.], [0., 0.], [2., -2.]])

        self.assertTrue(np.all(P_gt == P))
