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
    cell = lfp.CellGeometry(x=np.array([[0.] * 2] * n_seg),
                            y=np.array([[0.] * 2] * n_seg),
                            z=np.array([[1. * x, 1. * (x + 1)]
                                        for x in range(n_seg)]),
                            d=np.array([1.] * n_seg))
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

        self.assertTrue(np.all(imem == M @ imem))

    def test_CurrentDipoleMoment_00(self):
        '''test CurrentDipoleMoment'''
        cell = get_cell(n_seg=3)
        cdm = lfp.CurrentDipoleMoment(cell)
        M = cdm.get_response_matrix()

        imem = np.array([[-1., 1.],
                         [0., 0.],
                         [1., -1.]])

        P = M @ imem

        P_gt = np.array([[0., 0.], [0., 0.], [2., -2.]])

        self.assertTrue(np.all(P_gt == P))

    def test_PointSoucePotential_00(self):
        '''test PointSourcePotential implementation'''
        cell = get_cell(n_seg=3)
        sigma = 0.3
        r = np.array([[29.41099547, 43.06748789, -13.90864482, -40.44348899,
                       -29.4355596, -49.53871099, -33.70179906, 19.71508449,
                       -25.38725944, 42.52608652],
                      [24.36798674, -31.25870899, 22.6071361, -5.89078286,
                       44.44040172, 48.8092616, -34.2306679, -12.08847587,
                       -30.36317994, 30.79944143],
                      [-33.67242089, -9.71721014, 22.74564354, -27.14076556,
                       10.26397085, 30.1274518, -36.71772572, -22.21636375,
                       35.62274778, 41.273182]])
        psp = lfp.PointSourcePotential(cell=cell, x=r[0], y=r[1, ], z=r[2, ],
                                       sigma=sigma)
        M = psp.get_response_matrix()

        imem = np.array([[-1., 1.],
                         [0., 0.],
                         [1., -1.]])

        V_ex = M @ imem

        r_norm = np.empty((r.shape[1], cell.totnsegs))
        for i, (x, y, z) in enumerate(zip(cell.x.mean(axis=-1),
                                          cell.y.mean(axis=-1),
                                          cell.z.mean(axis=-1))):
            r_norm[:, i] = np.linalg.norm((r.T - np.r_[x, y, z]).T, axis=0)

        V_gt = (1 / (4 * np.pi * sigma * r_norm)) @ imem

        self.assertTrue(np.all(V_ex == V_gt))

    def test_PointSoucePotential_01(self):
        '''test PointSourcePotential implementation, when contact is within
        d/2 distance to segment'''
        cell = get_cell(n_seg=1)
        cell.d = np.array([2.])
        sigma = 0.3
        r = np.array([[0.1254235242],
                      [0.],
                      [cell.z.mean()]])
        psp = lfp.PointSourcePotential(cell=cell, x=r[0], y=r[1, ], z=r[2, ],
                                       sigma=sigma)
        M = psp.get_response_matrix()

        imem = np.array([[0., 1., -1.]]) * (4 * np.pi * sigma * cell.d[0] / 2)

        V_ex = M @ imem

        V_gt = np.array([[0., 1., -1.]])

        np.testing.assert_allclose(V_ex, V_gt)

    def test_PointSoucePotential_02(self):
        '''test PointSourcePotential implementation, when contact is at
        d/2 distance to segment'''
        cell = get_cell(n_seg=1)
        cell.d = np.array([2.])
        sigma = 0.3
        r = np.array([[cell.d[0] / 2],
                      [0.],
                      [cell.z.mean()]])
        psp = lfp.PointSourcePotential(cell=cell, x=r[0], y=r[1, ], z=r[2, ],
                                       sigma=sigma)
        M = psp.get_response_matrix()

        imem = np.array([[0., 1., -1.]]) * (4 * np.pi * sigma * cell.d[0] / 2)

        V_ex = M @ imem

        V_gt = np.array([[0., 1., -1.]])

        np.testing.assert_allclose(V_ex, V_gt)

    def test_LineSourcePotential_00(self):
        '''test LineSourcePotential implementation'''
        cell = get_cell(n_seg=1)
        cell.z = cell.z * 10

        lsp = lfp.LineSourcePotential(cell=cell,
                                      x=np.array([2.]),
                                      y=np.array([0.]),
                                      z=np.array([11.]),
                                      sigma=0.3)
        M = lsp.get_response_matrix()

        imem = np.array([[0., 1., -1.]])

        V_ex = M @ imem

        Deltas_n = 10.
        h_n = -11.
        r_n = 2.
        l_n = Deltas_n + h_n

        # Use Eq. C.13 case I (h<0, l<0) from Gary Holt's 1998 thesis
        V_gt = imem / (4 * np.pi * lsp.sigma * Deltas_n) * np.log(
            (np.sqrt(h_n**2 + r_n**2) - h_n)
            / (np.sqrt(l_n**2 + r_n**2) - l_n))

        np.testing.assert_allclose(V_ex, V_gt)

    def test_LineSourcePotential_01(self):
        '''test LineSourcePotential implementation'''
        cell = get_cell(n_seg=1)
        cell.z = cell.z * 10

        lsp = lfp.LineSourcePotential(cell=cell,
                                      x=np.array([2.]),
                                      y=np.array([0.]),
                                      z=np.array([5.]),
                                      sigma=0.3)
        M = lsp.get_response_matrix()

        imem = np.array([[0., 1., -1.]])

        V_ex = M @ imem

        Deltas_n = 10.
        h_n = -5.
        r_n = 2.
        l_n = Deltas_n + h_n

        # Use Eq. C.13 case II (h<0, l>0) from Gary Holt's 1998 thesis
        V_gt = imem / (4 * np.pi * lsp.sigma * Deltas_n) * np.log(
            (np.sqrt(h_n**2 + r_n**2) - h_n)
            * (np.sqrt(l_n**2 + r_n**2) + l_n)
            / r_n**2)

        np.testing.assert_allclose(V_ex, V_gt)

    def test_LineSourcePotential_02(self):
        '''test LineSourcePotential implementation when assigning
        a location inside cylindric volume'''
        cell = get_cell(n_seg=1)
        cell.z = cell.z * 10

        lsp = lfp.LineSourcePotential(cell=cell,
                                      x=np.array([.131441]),
                                      y=np.array([0.]),
                                      z=np.array([5.]),
                                      sigma=0.3)
        M = lsp.get_response_matrix()

        imem = np.array([[0., 1., -1.]])

        V_ex = M @ imem

        Deltas_n = 10.
        h_n = -5.
        r_n = cell.d[0] / 2
        l_n = Deltas_n + h_n

        # Use Eq. C.13 case II (h<0, l>0) from Gary Holt's 1998 thesis
        V_gt = imem / (4 * np.pi * lsp.sigma * Deltas_n) * np.log(
            (np.sqrt(h_n**2 + r_n**2) - h_n)
            * (np.sqrt(l_n**2 + r_n**2) + l_n)
            / r_n**2)

        np.testing.assert_allclose(V_ex, V_gt)

    def test_LineSourcePotential_03(self):
        '''test LineSourcePotential implementation'''
        cell = get_cell(n_seg=1)
        cell.z = cell.z * 10

        lsp = lfp.LineSourcePotential(cell=cell,
                                      x=np.array([2.]),
                                      y=np.array([0.]),
                                      z=np.array([-1.]),
                                      sigma=0.3)
        M = lsp.get_response_matrix()

        imem = np.array([[0., 1., -1.]])

        V_ex = M @ imem

        Deltas_n = 10.
        h_n = 1.
        r_n = 2.
        l_n = Deltas_n + h_n

        # Use Eq. C.13 case III (h>0, l>0) from Gary Holt's 1998 thesis
        V_gt = imem / (4 * np.pi * lsp.sigma * Deltas_n) * np.log(
            (np.sqrt(l_n**2 + r_n**2) + l_n)
            / (np.sqrt(h_n**2 + r_n**2) + h_n))

        np.testing.assert_allclose(V_ex, V_gt)

    def test_RecExtElectrode_00(self):
        """test RecExcElectrode implementation,
        method='pointsource'"""
        cell = get_cell(n_seg=1)
        sigma = 0.3
        r = np.array([[cell.d[0]],
                      [0.],
                      [cell.z.mean()]])
        el = lfp.RecExtElectrode(cell=cell,
                                 x=r[0], y=r[1, ], z=r[2, ],
                                 sigma=sigma,
                                 method='pointsource')
        M = el.get_response_matrix()

        imem = np.array([[0., 1., -1.]]) * (4 * np.pi * sigma * cell.d[0])

        V_ex = M @ imem

        V_gt = np.array([[0., 1., -1.]])

        np.testing.assert_allclose(V_ex, V_gt)

    def test_RecExtElectrode_01(self):
        """test LineSourcePotential implementation,
        method='linesource'"""
        cell = get_cell(n_seg=1)
        cell.z = cell.z * 10

        el = lfp.RecExtElectrode(cell=cell,
                                 x=np.array([2.]),
                                 y=np.array([0.]),
                                 z=np.array([5.]),
                                 sigma=0.3,
                                 method='linesource')
        M = el.get_response_matrix()

        imem = np.array([[0., 1., -1.]])

        V_ex = M @ imem

        Deltas_n = 10.
        h_n = -5.
        r_n = 2.
        l_n = Deltas_n + h_n

        # Use Eq. C.13 case II (h<0, l>0) from Gary Holt's 1998 thesis
        V_gt = imem / (4 * np.pi * el.sigma * Deltas_n) * np.log(
            (np.sqrt(h_n**2 + r_n**2) - h_n)
            * (np.sqrt(l_n**2 + r_n**2) + l_n)
            / r_n**2)

        np.testing.assert_allclose(V_ex, V_gt)

    def test_RecExtElectrode_02(self):
        """test RecExcElectrode implementation,
        method='root_as_point'"""
        cell = get_cell(n_seg=1)
        sigma = 0.3
        r = np.array([[cell.d[0]],
                      [0.],
                      [cell.z.mean()]])
        el = lfp.RecExtElectrode(cell=cell,
                                 x=r[0], y=r[1, ], z=r[2, ],
                                 sigma=sigma,
                                 method='root_as_point')
        M = el.get_response_matrix()

        imem = np.array([[0., 1., -1.]]) * (4 * np.pi * sigma * cell.d[0])

        V_ex = M @ imem

        V_gt = np.array([[0., 1., -1.]])

        np.testing.assert_allclose(V_ex, V_gt)
