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
import lfpykit as lfp


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
        M = lm.get_transformation_matrix()

        imem = np.arange(24).reshape((4, -1))

        self.assertTrue(np.all(imem == M @ imem))

    def test_CurrentDipoleMoment_00(self):
        '''test CurrentDipoleMoment'''
        cell = get_cell(n_seg=3)
        cdm = lfp.CurrentDipoleMoment(cell)
        M = cdm.get_transformation_matrix()

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
        M = psp.get_transformation_matrix()

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
        M = psp.get_transformation_matrix()

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
        M = psp.get_transformation_matrix()

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
        M = lsp.get_transformation_matrix()

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
        M = lsp.get_transformation_matrix()

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
        M = lsp.get_transformation_matrix()

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
        M = lsp.get_transformation_matrix()

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
        M = el.get_transformation_matrix()

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
        M = el.get_transformation_matrix()

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
        M = el.get_transformation_matrix()

        imem = np.array([[0., 1., -1.]]) * (4 * np.pi * sigma * cell.d[0])

        V_ex = M @ imem

        V_gt = np.array([[0., 1., -1.]])

        np.testing.assert_allclose(V_ex, V_gt)

    def test_RecExtElectrode_03(self):
        """test RecExcElectrode implementation,
        method='pointsource' with anisotropy"""
        cell = get_cell(n_seg=1)
        cell.x = np.array([[0., 2.4]])
        cell.y = np.array([[0., 2.4]])
        cell.z = np.array([[0., 2.4]])

        sigma = [0.6, 0.3, 0.45]
        r = np.array([[0.], [0.], [0.]])
        el = lfp.RecExtElectrode(cell=cell,
                                 x=r[0], y=r[1], z=r[2],
                                 sigma=sigma,
                                 method='pointsource')
        M = el.get_transformation_matrix()

        imem = np.array([[0., 1., -1.]])

        V_ex = M @ imem

        sigma_r = np.sqrt(sigma[1] * sigma[2] * 1.2**2
                          + sigma[0] * sigma[2] * 1.2**2
                          + sigma[0] * sigma[1] * 1.2**2)
        V_gt = np.array([[0., 1., -1.]]) / (4 * np.pi * sigma_r)

        np.testing.assert_allclose(V_ex, V_gt)

    def test_RecExtElectrode_04(self):
        """test RecExcElectrode implementation,
        method='root_as_point' and rootinds parameter"""
        cell = get_cell(n_seg=4)
        sigma = 0.3
        r = np.array([1, 0, 2])
        # all point sources
        el0 = lfp.PointSourcePotential(cell=cell,
                                       x=np.array([r[0]]),
                                       y=np.array([r[1]]),
                                       z=np.array([r[2]]),
                                       sigma=sigma)
        M0 = el0.get_transformation_matrix()

        # all line sources
        el1 = lfp.LineSourcePotential(cell=cell,
                                      x=np.array([r[0]]),
                                      y=np.array([r[1]]),
                                      z=np.array([r[2]]),
                                      sigma=sigma)
        M1 = el1.get_transformation_matrix()

        # vary which index is treated as point
        ids = np.arange(cell.totnsegs)
        for i in range(cell.totnsegs):
            el = lfp.RecExtElectrode(cell=cell,
                                     x=r[0],
                                     y=r[1],
                                     z=r[2],
                                     sigma=sigma,
                                     method='root_as_point',
                                     rootinds=np.array([i]))
            M = el.get_transformation_matrix()

            np.testing.assert_almost_equal(M0[0, i], M[0, i])
            np.testing.assert_equal(M1[0, ids != i], M[0, ids != i])

        # multiple roots
        for i in range(cell.totnsegs - 1):
            rootinds = np.array([i, i + 1])
            notroots = np.ones(cell.totnsegs, dtype=bool)
            notroots[rootinds] = False
            el = lfp.RecExtElectrode(cell=cell,
                                     x=r[0], y=r[1], z=r[2],
                                     sigma=sigma,
                                     method='root_as_point',
                                     rootinds=rootinds)
            M = el.get_transformation_matrix()

            np.testing.assert_allclose(M0[0, rootinds], M[0, rootinds])
            np.testing.assert_equal(M1[0, notroots], M[0, notroots])

    def test_RecMEAElectrode_00(self):
        """test RecMEAElectrode implementation,
        method='pointsource' with same conductivity in all layers"""
        cell = get_cell(n_seg=1)
        sigma = 0.3
        r = np.array([[0.],
                      [0.],
                      [0.]])
        mea = lfp.RecMEAElectrode(cell=cell,
                                  x=r[0], y=r[1, ], z=r[2, ],
                                  sigma_T=sigma, sigma_S=sigma, sigma_G=sigma,
                                  method='pointsource')
        M = mea.get_transformation_matrix()

        imem = np.array([[0., 1., -1.]]) * (4 * np.pi * sigma * cell.z.mean())

        V_ex = M @ imem

        V_gt = np.array([[0., 1., -1.]])

        np.testing.assert_allclose(V_ex, V_gt)

    def test_RecMEAElectrode_01(self):
        '''test _test_cell_extent method w/wo. squeeze along z-axis'''
        electrodeParams = {
            'sigma_T': 0.3,
            'sigma_S': 1.5,
            'sigma_G': 0.0,
            'h': 200,
            'x': np.linspace(0, 1000, 11),
            'y': np.zeros(11),
            'z': np.zeros(11),
            'method': "pointsource",
            'squeeze_cell_factor': None,
            'verbose': False
        }

        stick = lfp.CellGeometry(x=np.array([[-17.5, 17.5],
                                             [0., 23.],
                                             [23., 46.],
                                             [46., 69.],
                                             [69., 92.],
                                             [92., 115.],
                                             [115., 125.],
                                             [125., 135.],
                                             [135., 145.],
                                             [145., 155.],
                                             [155., 165.],
                                             [115., 118.33333333],
                                             [118.33333333, 121.66666667],
                                             [121.66666667, 125.]]),
                                 y=np.array([[0., 0.],
                                             [0., 0.],
                                             [0., 0.],
                                             [0., 0.],
                                             [0., 0.],
                                             [0., 0.],
                                             [0., 4.],
                                             [4., 8.],
                                             [8., 12.],
                                             [12., 16.],
                                             [16., 20.],
                                             [0., 0.],
                                             [0., 0.],
                                             [0., 0.]]),
                                 z=np.array([[0, 0],
                                             [0, 0],
                                             [0, 0],
                                             [0, 0],
                                             [0, 0],
                                             [0, 0],
                                             [0, 0],
                                             [10, 20],
                                             [20, 30],
                                             [30, 40],
                                             [40, 50],
                                             [0, -10],
                                             [-10, -20],
                                             [-20, -30]]),
                                 d=np.zeros(14))

        stick.z = stick.z + 1
        MEA = lfp.RecMEAElectrode(stick, **electrodeParams)
        np.testing.assert_raises(RuntimeError, MEA._test_cell_extent)

        stick.z = stick.z + 198.
        MEA = lfp.RecMEAElectrode(stick, **electrodeParams)
        np.testing.assert_raises(RuntimeError, MEA._test_cell_extent)

        electrodeParams = {
            'sigma_T': 0.3,
            'sigma_S': 1.5,
            'sigma_G': 0.0,
            'h': 200,
            'x': np.linspace(0, 1000, 11),
            'y': np.zeros(11),
            'z': np.zeros(11),
            'method': "pointsource",
            'squeeze_cell_factor': 0.1,
        }

        stick.z = stick.z - 200
        MEA = lfp.RecMEAElectrode(stick, **electrodeParams)
        np.testing.assert_raises(RuntimeError, MEA._test_cell_extent)

        stick.z = np.zeros_like(stick.z) + 202
        MEA = lfp.RecMEAElectrode(stick, **electrodeParams)
        np.testing.assert_raises(RuntimeError, MEA._test_cell_extent)

    def test_RecMEAElectrode_02(self):
        '''test return_comp_outside_slice'''
        electrodeParams = {
            'sigma_T': 0.3,
            'sigma_S': 1.5,
            'sigma_G': 0.0,
            'h': 200,
            'x': np.linspace(0, 1000, 11),
            'y': np.zeros(11),
            'z': np.zeros(11),
            'method': "pointsource",
            'squeeze_cell_factor': None,
        }

        stick = lfp.CellGeometry(x=np.array([[-17.5, 17.5],
                                             [0., 23.],
                                             [23., 46.],
                                             [46., 69.],
                                             [69., 92.],
                                             [92., 115.],
                                             [115., 125.],
                                             [125., 135.],
                                             [135., 145.],
                                             [145., 155.],
                                             [155., 165.],
                                             [115., 118.33333333],
                                             [118.33333333, 121.66666667],
                                             [121.66666667, 125.]]),
                                 y=np.array([[0., 0.],
                                             [0., 0.],
                                             [0., 0.],
                                             [0., 0.],
                                             [0., 0.],
                                             [0., 0.],
                                             [0., 4.],
                                             [4., 8.],
                                             [8., 12.],
                                             [12., 16.],
                                             [16., 20.],
                                             [0., 0.],
                                             [0., 0.],
                                             [0., 0.]]),
                                 z=np.array([[0, 0],
                                             [0, 0],
                                             [0, 0],
                                             [0, 0],
                                             [0, 0],
                                             [0, 0],
                                             [0, 0],
                                             [10, 20],
                                             [20, 30],
                                             [30, 40],
                                             [40, 50],
                                             [0, -10],
                                             [-10, -20],
                                             [-20, -30]]),
                                 d=np.zeros(14))
        stick.z = stick.z + 100
        MEA = lfp.RecMEAElectrode(stick, **electrodeParams)
        np.testing.assert_raises(RuntimeError, MEA._return_comp_outside_slice)
        true_bad_comp = np.array([2, 3, 6])

        stick.z[true_bad_comp, 0] = 1000
        bad_comp, reason = MEA._return_comp_outside_slice()
        np.testing.assert_equal(reason, "zstart above")
        np.testing.assert_equal(true_bad_comp, bad_comp)
        stick.z[true_bad_comp, 0] = 100

        stick.z[true_bad_comp, 0] = -1000
        bad_comp, reason = MEA._return_comp_outside_slice()
        np.testing.assert_equal(reason, "zstart below")
        np.testing.assert_equal(true_bad_comp, bad_comp)
        stick.z[true_bad_comp, 0] = 100

        stick.z[true_bad_comp, -1] = 1000
        bad_comp, reason = MEA._return_comp_outside_slice()
        np.testing.assert_equal(reason, "zend above")
        np.testing.assert_equal(true_bad_comp, bad_comp)
        stick.z[true_bad_comp, -1] = 100

        stick.z[true_bad_comp, -1] = -1000
        bad_comp, reason = MEA._return_comp_outside_slice()
        np.testing.assert_equal(reason, "zend below")
        np.testing.assert_equal(true_bad_comp, bad_comp)
        stick.z[true_bad_comp, -1] = 100

    def test_RecMEAElectrode_03(self):
        '''test_position_shifted_slice'''
        electrodeParams = {
            'sigma_T': 0.3,
            'sigma_S': 1.5,
            'sigma_G': 0.0,
            'h': 200,
            'z_shift': -200,
            'x': np.linspace(0, 1000, 11),
            'y': np.zeros(11),
            'z': np.zeros(11) - 100,
            'method': "pointsource",
            'squeeze_cell_factor': None,
        }

        stick = lfp.CellGeometry(x=np.array([[-17.5, 17.5],
                                             [0., 23.],
                                             [23., 46.],
                                             [46., 69.],
                                             [69., 92.],
                                             [92., 115.],
                                             [115., 125.],
                                             [125., 135.],
                                             [135., 145.],
                                             [145., 155.],
                                             [155., 165.],
                                             [115., 118.33333333],
                                             [118.33333333, 121.66666667],
                                             [121.66666667, 125.]]),
                                 y=np.array([[0., 0.],
                                             [0., 0.],
                                             [0., 0.],
                                             [0., 0.],
                                             [0., 0.],
                                             [0., 0.],
                                             [0., 4.],
                                             [4., 8.],
                                             [8., 12.],
                                             [12., 16.],
                                             [16., 20.],
                                             [0., 0.],
                                             [0., 0.],
                                             [0., 0.]]),
                                 z=np.array([[0, 0],
                                             [0, 0],
                                             [0, 0],
                                             [0, 0],
                                             [0, 0],
                                             [0, 0],
                                             [0, 0],
                                             [10, 20],
                                             [20, 30],
                                             [30, 40],
                                             [40, 50],
                                             [0, -10],
                                             [-10, -20],
                                             [-20, -30]]),
                                 d=np.zeros(14))
        stick.z = stick.z - 100

        MEA = lfp.RecMEAElectrode(stick, **electrodeParams)
        MEA._test_cell_extent()

    def test_LaminarCurrentSourceDensity_00(self):
        '''test LaminarCurrentSourceDensity implementation
        2 segs, 2 volumes, each segment in each volume'''
        cell = get_cell(n_seg=3)
        cell.z = cell.z * 10
        cell._set_length()

        h = 10.
        r = 10.
        V = np.pi * r**2 * h
        M_gt = np.array([[1., 0., 0.],
                         [0., 1., 0.]]) / V

        csd = lfp.LaminarCurrentSourceDensity(
            cell=cell,
            z=np.array([[i * h, (i + 1) * h] for i in range(2)]),
            r=np.array([r, r]))
        M = csd.get_transformation_matrix()

        np.testing.assert_allclose(M_gt, M)

    def test_LaminarCurrentSourceDensity_01(self):
        '''test LaminarCurrentSourceDensity implementation'''
        cell = get_cell(n_seg=4)
        cell.z = cell.z * 10
        cell.z -= 10
        cell._set_length()

        h = 10.
        r = 10.
        V = np.pi * r**2 * h
        M_gt = np.array([[0., 1., 0., 0.],
                         [0., 0., 1., 0.]]) / V

        csd = lfp.LaminarCurrentSourceDensity(
            cell=cell,
            z=np.array([[i * h, (i + 1) * h] for i in range(2)]),
            r=np.array([r, r]))
        M = csd.get_transformation_matrix()

        np.testing.assert_allclose(M_gt, M)

    def test_LaminarCurrentSourceDensity_02(self):
        '''test LaminarCurrentSourceDensity implementation'''
        cell = get_cell(n_seg=4)
        cell.z = cell.z * 10
        cell.z -= 5
        cell._set_length()

        h = 10.
        r = 10.
        V = np.pi * r**2 * h
        M_gt = np.array([[0.5, 0.5, 0., 0.],
                         [0., 0.5, 0.5, 0.]]) / V

        csd = lfp.LaminarCurrentSourceDensity(
            cell=cell,
            z=np.array([[i * h, (i + 1) * h] for i in range(2)]),
            r=np.array([r, r]))
        M = csd.get_transformation_matrix()

        np.testing.assert_allclose(M_gt, M)

    def test_LaminarCurrentSourceDensity_3(self):
        '''test LaminarCurrentSourceDensity implementation'''
        cell = get_cell(n_seg=3)
        cell.z = cell.z * 10

        cell.z -= .1
        cell._set_length()

        h = 10.
        r = 10.
        V = np.pi * r**2 * h
        M_gt = np.array([[0.99, 0.01, 0.],
                         [0.0, 0.99, 0.01]]) / V

        csd = lfp.LaminarCurrentSourceDensity(
            cell=cell,
            z=np.array([[i * h, (i + 1) * h] for i in range(2)]),
            r=np.array([r, r]))
        M = csd.get_transformation_matrix()

        np.testing.assert_allclose(M_gt, M)

    def test_LaminarCurrentSourceDensity_4(self):
        '''test LaminarCurrentSourceDensity implementation'''
        cell = get_cell(n_seg=4)
        cell.z = cell.z * 10
        cell.z += 2.5
        cell._set_length()

        h = 10.
        r = 10.
        V = np.pi * r**2 * h
        M_gt = np.array([[0.75, 0.0, 0., 0.],
                         [0.25, 0.75, 0., 0.]]) / V

        csd = lfp.LaminarCurrentSourceDensity(
            cell=cell,
            z=np.array([[i * h, (i + 1) * h] for i in range(2)]),
            r=np.array([r, r]))
        M = csd.get_transformation_matrix()

        np.testing.assert_allclose(M_gt, M)

    def test_LaminarCurrentSourceDensity_05(self):
        '''test LaminarCurrentSourceDensity implementation

        lateral offset segmens
        '''
        cell = get_cell(n_seg=4)
        cell.z = cell.z * 10
        cell.x[1, 1] += 20
        cell.x[2:, ] += 20
        cell._set_length()

        h = 10.
        r = 10.
        V = np.pi * r**2 * h
        M_gt = np.array([[1., 0., 0., 0.],
                         [0., 0.5, 0., 0.]]) / V

        csd = lfp.LaminarCurrentSourceDensity(
            cell=cell,
            z=np.array([[i * h, (i + 1) * h] for i in range(2)]),
            r=np.array([r, r]))
        M = csd.get_transformation_matrix()

        np.testing.assert_allclose(M_gt, M)

    def test_LaminarCurrentSourceDensity_06(self):
        '''test LaminarCurrentSourceDensity implementation

        lateral and vertical offset of segments
        '''
        cell = get_cell(n_seg=4)
        cell.z = cell.z * 10
        cell.z += 5.
        cell.x[1, 1] += 20
        cell.x[2:, ] += 20
        cell._set_length()

        h = 10.
        r = 10.
        V = np.pi * r**2 * h
        M_gt = np.array([[0.5, 0., 0., 0.],
                         [0.5, 0.5, 0., 0.]]) / V

        csd = lfp.LaminarCurrentSourceDensity(
            cell=cell,
            z=np.array([[i * h, (i + 1) * h] for i in range(2)]),
            r=np.array([r, r]))
        M = csd.get_transformation_matrix()

        np.testing.assert_allclose(M_gt, M)

    def test_VolumetricCurrentSourceDensity_00(self):
        cell = get_cell(n_seg=1)
        cell.z = cell.z * 10.
        cell._set_length()
        csd = lfp.VolumetricCurrentSourceDensity(cell,
                                                 x=np.linspace(-5., 5., 2),
                                                 y=np.linspace(-5., 5., 2),
                                                 z=np.linspace(-5., 5., 2),
                                                 dl=1.)
        M = csd.get_transformation_matrix()

        M_gt = np.zeros((1, 1, 1, 1))
        M_gt[0, 0, 0, 0] = 0.5

        np.testing.assert_allclose(M_gt, M)

    def test_VolumetricCurrentSourceDensity_01(self):
        cell = get_cell(n_seg=1)
        cell.z = cell.z * 10.
        cell._set_length()
        csd = lfp.VolumetricCurrentSourceDensity(cell,
                                                 x=np.linspace(-5., 5., 2),
                                                 y=np.linspace(-5., 5., 2),
                                                 z=np.linspace(-5., 15., 3),
                                                 dl=1.)
        M = csd.get_transformation_matrix()

        M_gt = np.zeros((1, 1, 2, 1))
        M_gt[0, 0, :, 0] = 0.5

        np.testing.assert_allclose(M_gt, M)

    def test_VolumetricCurrentSourceDensity_02(self):
        cell = get_cell(n_seg=1)
        cell.z = cell.z * 10.
        cell._set_length()
        csd = lfp.VolumetricCurrentSourceDensity(cell,
                                                 x=np.linspace(-5., 5., 2),
                                                 y=np.linspace(-5., 5., 2),
                                                 z=np.linspace(-15., 25., 5),
                                                 dl=1.)
        M = csd.get_transformation_matrix()

        M_gt = np.zeros((1, 1, 4, 1))
        M_gt[0, 0, 1:3, 0] = 0.5

        np.testing.assert_allclose(M_gt, M)

    def test_VolumetricCurrentSourceDensity_03(self):
        cell = get_cell(n_seg=2)
        cell.z -= 1.
        cell.z = cell.z * 10.
        cell._set_length()
        csd = lfp.VolumetricCurrentSourceDensity(cell,
                                                 x=np.linspace(-5., 5., 2),
                                                 y=np.linspace(-5., 5., 2),
                                                 z=np.linspace(-20., 20., 5),
                                                 dl=1.)
        M = csd.get_transformation_matrix()

        M_gt = np.zeros((1, 1, 4, 2))
        M_gt[0, 0, 1, 0] = 1.
        M_gt[0, 0, 2, 1] = 1.

        np.testing.assert_allclose(M_gt, M)

    def test_VolumetricCurrentSourceDensity_04(self):
        cell = get_cell(n_seg=2)
        cell.z -= 0.5
        cell.z = cell.z * 10.
        cell._set_length()
        csd = lfp.VolumetricCurrentSourceDensity(cell,
                                                 x=np.linspace(-5., 5., 2),
                                                 y=np.linspace(-5., 5., 2),
                                                 z=np.linspace(-20., 20., 5),
                                                 dl=1.)
        M = csd.get_transformation_matrix()

        M_gt = np.zeros((1, 1, 4, 2))
        M_gt[0, 0, 1:3, 0] = 0.5
        M_gt[0, 0, 2:4, 1] = 0.5

        np.testing.assert_allclose(M_gt, M)
