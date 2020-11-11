#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Collection of classes defining forward models applicable with current dipole
moment predictions.

Copyright (C) 2017 Computational Neuroscience Group, NMBU.
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
"""
import os
import sys
import math
from scipy.special import lpmv
import numpy as np
from warnings import warn


class FourSphereVolumeConductor(object):
    """
    Main class for computing extracellular potentials in a four-sphere
    volume conductor model that assumes homogeneous, isotropic, linear
    (frequency independent) conductivity within the inner sphere and outer
    shells. The conductance outside the outer shell is 0 (air).

    This class implements the corrected 4-sphere model described in [1]_, [2]_

    References
    ----------
    .. [1] Næss S, Chintaluri C, Ness TV, Dale AM, Einevoll GT and Wójcik DK
        (2017) Corrected Four-sphere Head Model for EEG Signals. Front. Hum.
        Neurosci. 11:490. doi: 10.3389/fnhum.2017.00490
    .. [2] Hagen E, Næss S, Ness TV and Einevoll GT (2018) Multimodal Modeling
        of Neural Network Activity: Computing LFP, ECoG, EEG, and MEG Signals
        With LFPy 2.0. Front. Neuroinform. 12:92. doi: 10.3389/fninf.2018.00092

    See also
    --------
    InfiniteVolumeConductor
    MEG

    Parameters
    ----------
    r_electrodes: ndarray, dtype=float
        Shape (n_contacts, 3) array containing n_contacts electrode locations
        in cartesian coordinates in units of [µm].
        All ``r_el`` in ``r_electrodes`` must be less than or equal to scalp
        radius and larger than the distance between dipole and sphere
        center: ``|rz| < r_el <= radii[3]``.
    radii: list, dtype=float
        Len 4 list with the outer radii in units of [µm] for the 4
        concentric shells in the four-sphere model: brain, csf, skull and
        scalp, respectively.
    sigmas: list, dtype=float
        Len 4 list with the electrical conductivity in units of [S/m] of
        the four shells in the four-sphere model: brain, csf, skull and
        scalp, respectively.
    iter_factor: float
        iteration-stop factor

    Examples
    --------
    Compute extracellular potential from current dipole moment in four-sphere
    head model:

    >>> from lfpykit.eegmegcalc import FourSphereVolumeConductor
    >>> import numpy as np
    >>> radii = [79000., 80000., 85000., 90000.]  # [µm]
    >>> sigmas = [0.3, 1.5, 0.015, 0.3]  # [S/m]
    >>> r_electrodes = np.array([[0., 0., 90000.], [0., 85000., 0.]]) # [µm]
    >>> sphere_model = FourSphereVolumeConductor(r_electrodes, radii,
    >>>                                          sigmas)
    >>> # current dipole moment
    >>> p = np.array([[10.]*10, [10.]*10, [10.]*10]) # 10 timesteps [nA µm]
    >>> dipole_location = np.array([0., 0., 78000.])  # [µm]
    >>> # compute potential
    >>> sphere_model.calc_potential(p, dipole_location)  # [mV]
    array([[1.06247669e-08, 1.06247669e-08, 1.06247669e-08, 1.06247669e-08,
            1.06247669e-08, 1.06247669e-08, 1.06247669e-08, 1.06247669e-08,
            1.06247669e-08, 1.06247669e-08],
           [2.39290752e-10, 2.39290752e-10, 2.39290752e-10, 2.39290752e-10,
            2.39290752e-10, 2.39290752e-10, 2.39290752e-10, 2.39290752e-10,
            2.39290752e-10, 2.39290752e-10]])
    """
    def __init__(self,
                 r_electrodes,
                 radii=[79000., 80000., 85000., 90000.],
                 sigmas=[0.3, 1.5, 0.015, 0.3],
                 iter_factor=2. / 99. * 1e-6):
        """Initialize class FourSphereVolumeConductor"""
        self.r1 = float(radii[0])
        self.r2 = float(radii[1])
        self.r3 = float(radii[2])
        self.r4 = float(radii[3])

        self.sigma1 = float(sigmas[0])
        self.sigma2 = float(sigmas[1])
        self.sigma3 = float(sigmas[2])
        self.sigma4 = float(sigmas[3])

        self.r12 = self.r1 / self.r2
        self.r21 = self.r2 / self.r1
        self.r23 = self.r2 / self.r3
        self.r32 = self.r3 / self.r2
        self.r34 = self.r3 / self.r4
        self.r43 = self.r4 / self.r3

        self.sigma12 = self.sigma1 / self.sigma2
        self.sigma21 = self.sigma2 / self.sigma1
        self.sigma23 = self.sigma2 / self.sigma3
        self.sigma32 = self.sigma3 / self.sigma2
        self.sigma34 = self.sigma3 / self.sigma4
        self.sigma43 = self.sigma4 / self.sigma3

        self.rxyz = r_electrodes
        self.r = np.sqrt(np.sum(r_electrodes ** 2, axis=1))

        self.iteration_stop_factor = iter_factor
        self._check_params()

    def _check_params(self):
        """Check that radii, sigmas and r contain reasonable values"""
        if (self.r1 < 0) or (self.r1 > self.r2) or (
                self.r2 > self.r3) or (self.r3 > self.r4):
            raise RuntimeError(
                'Radii of brain (radii[0]), CSF (radii[1]), '
                'skull (radii[2]) and scalp (radii[3]) '
                'must be positive ints or floats such that '
                '0 < radii[0] < radii[1] < radii[2] < radii[3].')

        if (self.sigma1 < 0) or (self.sigma2 < 0) or (
                self.sigma3 < 0) or (self.sigma4 < 0):
            raise RuntimeError('Conductivities (sigmas), must contain positive'
                               ' ints or floats.')

        if any(r > self.r4 for r in self.r):
            raise ValueError('Electrode located outside head model.'
                             'r > radii[3]. r = %s, r4 = %s',
                             self.r, self.r4)

    def _rz_params(self, rz):
        """Define dipole location vector, and check that dipole is located in
        the brain, closer to the center than any measurement location."""
        self._rzloc = rz
        self._rz = np.sqrt(np.sum(rz ** 2))
        with np.errstate(invalid='ignore'):
            self._z = self._rzloc / self._rz
        if self._rz == 0:
            raise RuntimeError('Placing dipole in center of head model '
                               'causes division by zero.')

        self._rz1 = self._rz / self.r1

        if self._rz1 >= 1:
            raise RuntimeError(
                'Dipole should be placed inside brain, i.e. |rz| < |r1|')

        elif self._rz1 > 0.99999:
            warn(
                'Dipole should be placed minimum ~1µm away from brain surface,'
                ' to avoid extremely slow convergence.')

        elif self._rz1 > 0.9999:
            warn('Computation time might be long due to slow convergence. '
                 'Can be avoided by placing dipole further away from '
                 'brain surface.')

        if any(r < self._rz for r in self.r):
            raise RuntimeError('Electrode must be farther away from '
                               'brain center than dipole: r > rz.'
                               'r = %s, rz = %s', self.r, self._rz)

        # compute theta angle between rzloc and rxyz
        self._theta = self._calc_theta()

    def get_dipole_potential(self, p, dipole_location):
        """
        Return electric potential from current dipole moment ``p`` in
        location ``dipole_location`` in  locations ``r_electrodes``

        Parameters
        ----------
        p: ndarray, dtype=float
            Shape (3, n_timesteps) array containing the x,y,z components of the
            current dipole moment in units of (nA*µm) for all timesteps.
        dipole_location: ndarray, dtype=float
            Shape (3, ) array containing the position of the current dipole in
            cartesian coordinates. Units of [µm].

        Returns
        -------
        potential: ndarray, dtype=float
            Shape (n_contacts, n_timesteps) array containing the electric
            potential at contact point(s) FourSphereVolumeConductor.r in units
            of [mV] for all timesteps of current dipole moment p.

        """

        self._rz_params(dipole_location)
        n_contacts = self.r.shape[0]
        n_timesteps = p.shape[1]

        if np.linalg.norm(p) != 0:
            p_rad, p_tan = self._decompose_dipole(p)
        else:
            p_rad = np.zeros((3, n_timesteps))
            p_tan = np.zeros((3, n_timesteps))
        if np.linalg.norm(p_rad) != 0.:
            pot_rad = self._calc_rad_potential(p_rad)
        else:
            pot_rad = np.zeros((n_contacts, n_timesteps))

        if np.linalg.norm(p_tan) != 0.:
            pot_tan = self._calc_tan_potential(p_tan)
        else:
            pot_tan = np.zeros((n_contacts, n_timesteps))

        pot_tot = pot_rad + pot_tan
        return pot_tot

    def get_transformation_matrix(self, dipole_location):
        '''
        Get linear response matrix mapping current dipole moment in [nA µm]
        located in location `rz` to extracellular potential in [mV]
        at recording sites `FourSphereVolumeConductor.` [µm]

        parameters
        ----------
        dipole_location: ndarray, dtype=float
            Shape (3, ) array containing the position of the current dipole in
            cartesian coordinates. Units of [µm].

        Returns
        -------
        response_matrix: ndarray
            shape (n_contacts, 3) ndarray
        '''
        return self.get_dipole_potential(np.eye(3), dipole_location)

    def _decompose_dipole(self, p):
        """
        Decompose current dipole moment vector in radial and tangential terms

        Parameters
        ----------
        p: ndarray, dtype=float
            Shape (3, n_timesteps) array containing the x,y,z-components of the
            current dipole moment in units of (nA*µm) for all timesteps

        Returns:
        -------
        p_rad: ndarray, dtype=float
            Shape (3, n_timesteps) array, radial part of p,
            parallel to self._rz
        p_tan: ndarray, dtype=float
            Shape (3, n_timesteps) array, tangential part of p,
            orthogonal to self._rz
        """
        z_ = np.expand_dims(self._z, -1)  # reshape z-axis vector
        p_rad = z_ @ (z_.T @ p)
        p_tan = p - p_rad
        return p_rad, p_tan

    def _calc_rad_potential(self, p_rad):
        """
        Return potential from radial dipole p_rad at location rz measured at r

        Parameters
        ----------
        p_rad: ndarray, dtype=float
            Shape (3, n_timesteps) array, radial part of p
            in units of (nA*µm), parallel to self._rz

        Returns
        -------
        potential: ndarray, dtype=float
            Shape (n_contacts, n_timesteps) array containing the extracecllular
            potential at n_contacts contact point(s)
            FourSphereVolumeConductor.r in units of [mV] for all timesteps
            of p_rad
        """

        p_tot = np.linalg.norm(p_rad, axis=0)
        s_vector = self._sign_rad_dipole(p_rad)
        phi_const = s_vector * p_tot / \
            (4 * np.pi * self.sigma1 * self._rz ** 2)
        n_terms = np.zeros((len(self.r), len(p_tot)))
        for el_point in range(len(self.r)):
            r_point = self.r[el_point]
            theta_point = self._theta[el_point]
            if r_point <= self.r1:
                n_terms[el_point] = self._potential_brain_rad(r_point,
                                                              theta_point)
            elif r_point <= self.r2:
                n_terms[el_point] = self._potential_csf_rad(r_point,
                                                            theta_point)
            elif r_point <= self.r3:
                n_terms[el_point] = self._potential_skull_rad(r_point,
                                                              theta_point)
            else:
                n_terms[el_point] = self._potential_scalp_rad(r_point,
                                                              theta_point)
        potential = phi_const * n_terms
        return potential

    def _calc_tan_potential(self, p_tan):
        """
        Return potential from tangential dipole P at location rz measured at r

        Parameters
        ----------
        p_tan: ndarray, dtype=float
            Shape (3, n_timesteps) array, tangential part of p
            in units of (nA*µm), orthogonal to self._rz

        Returns
        _______
        potential: ndarray, dtype=float
            Shape (n_contacts, n_timesteps) array containing the extracecllular
            potential at n_contacts contact point(s)
            FourSphereVolumeConductor.r in units of [mV] for all timesteps
            of p_tan
        """
        phi = self._calc_phi(p_tan)
        p_tot = np.linalg.norm(p_tan, axis=0)
        phi_hom = - p_tot / (4 * np.pi * self.sigma1 *
                             self._rz ** 2) * np.sin(phi)
        n_terms = np.zeros((len(self.r), 1))
        for el_point in range(len(self.r)):
            r_point = self.r[el_point]
            theta_point = self._theta[el_point]
            # if r_electrode is orthogonal to p_tan, i.e. theta = 0 or
            # theta = pi,  there is no contribution to electric potential
            # from p_tan
            if (theta_point == 0.) or (theta_point == np.pi):
                n_terms[el_point] = 0
            elif r_point <= self.r1:
                n_terms[el_point] = self._potential_brain_tan(
                    r_point, theta_point)
            elif r_point <= self.r2:
                n_terms[el_point] = self._potential_csf_tan(
                    r_point, theta_point)
            elif r_point <= self.r3:
                n_terms[el_point] = self._potential_skull_tan(
                    r_point, theta_point)
            else:
                n_terms[el_point] = self._potential_scalp_tan(
                    r_point, theta_point)
        potential = n_terms * phi_hom

        return potential

    def _calc_theta(self):
        """
        Return polar angle(s) between rzloc and contact point location(s)

        Returns
        -------
        theta: ndarray, dtype=float
            Shape (n_contacts, ) array containing polar angle
            in units of (radians) between z-axis and n_contacts contact
            point location vector(s) in FourSphereVolumeConductor.rxyz
            z-axis is defined in the direction of rzloc and the radial dipole.
        """
        cos_theta = (self.rxyz @ self._rzloc) / (
            np.linalg.norm(self.rxyz, axis=1) * np.linalg.norm(self._rzloc))
        theta = np.arccos(cos_theta)
        return theta

    def _calc_phi(self, p_tan):
        """
        Return azimuthal angle between x-axis and contact point locations(s)

        Parameters
        ----------
        p_tan: ndarray, dtype=float
            Shape (3, n_timesteps) array containing
            tangential component of current dipole moment in units of (nA*µm)

        Returns
        -------
        phi: ndarray, dtype=float
            Shape (n_contacts, n_timesteps) array containing azimuthal angle
            in units of (radians) between x-axis vector(s) and projection of
            contact point location vector(s) rxyz into xy-plane.
            z-axis is defined in the direction of rzloc.
            y-axis is defined in the direction of p_tan (orthogonal to rzloc).
            x-axis is defined as cross product between p_tan and rzloc (x).
        """

        # project rxyz onto z-axis (rzloc)
        proj_rxyz_rz = self.rxyz * self._z
        # find projection of rxyz in xy-plane
        rxy = self.rxyz - proj_rxyz_rz
        # define x-axis
        x = np.cross(p_tan.T, self._z)

        phi = np.zeros((len(self.rxyz), p_tan.shape[1]))
        # create masks to avoid computing phi when phi is not defined
        mask = np.ones(phi.shape, dtype=bool)
        # phi is not defined when theta= 0,pi or |p_tan| = 0
        mask[(self._theta == 0) | (self._theta == np.pi)] = np.zeros(
            p_tan.shape[1])
        mask[:, np.abs(np.linalg.norm(p_tan, axis=0)) == 0] = 0

        cos_phi = np.zeros(phi.shape)
        # compute cos_phi using mask to avoid zerodivision
        cos_phi[mask] = (rxy @ x.T)[mask] \
            / np.outer(np.linalg.norm(rxy, axis=1),
                       np.linalg.norm(x, axis=1))[mask]

        # compute phi in [0, pi]
        phi[mask] = np.arccos(cos_phi[mask])

        # nb: phi in [-pi, pi]. since p_tan defines direction of y-axis,
        # phi < 0 when rxy*p_tan < 0
        phi[(rxy @ p_tan) < 0] *= -1

        return phi

    def _sign_rad_dipole(self, p):
        """
        Determine whether radial dipoles are pointing inwards or outwards

        Parameters
        ----------
        p: ndarray, dtype=float
            Shape (3, n_timesteps) array containing the current dipole moment
             in cartesian coordinates for all n_timesteps in units of (nA*µm)

        Returns
        -------
        sign_vector: ndarray
            Shape (n_timesteps, ) array containing +/-1 for all
            current dipole moments in p.
            If radial part of p[i] points outwards, sign_vector[i] = 1.
            If radial part of p[i] points inwards, sign_vector[i] = -1.

        """
        return np.sign(self._rzloc @ p)

    def _potential_brain_rad(self, r, theta):
        """
        Return factor for calculation of potential in brain from rad. dipole

        Parameters
        ----------
        r: float
            Distance from origin to brain electrode location in units of [µm]
        theta: float
            Polar angle between brain electrode location and
            dipole location vector rzloc in units of (radians)

        Returns
        -------
        pot_sum: float
            Summationfactor for calculation of electrical potential in brain
            from radial current dipole moment. (unitless)
        """
        n = 1
        const = 1.
        coeff_sum = 0.
        consts = []
        while const > self.iteration_stop_factor * 1e-6 * coeff_sum:
            c1n = self._calc_c1n(n)
            const = n * (c1n * (r / self.r1) ** n + (self._rz / r) ** (n + 1))
            coeff_sum += const
            consts.append(const)
            n += 1
        consts = np.insert(consts, 0, 0)  # legendre function starts with P0
        leg_consts = np.polynomial.legendre.Legendre(consts)
        pot_sum = leg_consts(np.cos(theta))
        return pot_sum

    def _potential_csf_rad(self, r, theta):
        """
        Return factor for calculation of potential in CSF from rad. dipole

        Parameters
        ----------
        r: float
            Distance from origin to CSF electrode location in units of [µm]
        theta: float
            Polar angle between CSF electrode location and
            dipole location vector rzloc in units of (radians)

        Returns
        -------
        pot_sum: float
            Summation factor for calculation of electrical potential in CSF
            from radial current dipole moment. (unitless)
        """
        n = 1
        const = 1.
        coeff_sum = 0.
        consts = []
        while const > self.iteration_stop_factor * coeff_sum:
            term1 = self._calc_csf_term1(n, r)
            term2 = self._calc_csf_term2(n, r)
            const = n * (term1 + term2)
            coeff_sum += const
            consts.append(const)
            n += 1
        # since the legendre function starts with P0
        consts = np.insert(consts, 0, 0)
        leg_consts = np.polynomial.legendre.Legendre(consts)
        pot_sum = leg_consts(np.cos(theta))
        return pot_sum

    def _potential_skull_rad(self, r, theta):
        """
        Return factor for calculation of potential in skull from rad. dipole

        Parameters
        ----------
        r: float
            Distance from origin to skull electrode location in units of [µm]
        theta: float
            Polar angle between skull electrode location and
            dipole location vector rzloc in units of (radians)

        Returns
        -------
        pot_sum: float
            Summation factor for calculation of electrical potential in skull
            from radial current dipole moment. (unitless)
        """
        n = 1
        const = 1.
        coeff_sum = 0.
        consts = []
        while const > self.iteration_stop_factor * coeff_sum:
            c3n = self._calc_c3n(n)
            d3n = self._calc_d3n(n, c3n)
            const = n * (c3n * (r / self.r3) ** n +
                         d3n * (self.r3 / r) ** (n + 1))
            coeff_sum += const
            consts.append(const)
            n += 1
        # since the legendre function starts with P0
        consts = np.insert(consts, 0, 0)
        leg_consts = np.polynomial.legendre.Legendre(consts)
        pot_sum = leg_consts(np.cos(theta))
        return pot_sum

    def _potential_scalp_rad(self, r, theta):
        """
        Return factor for calculation of potential in scalp from radial dipole

        Parameters
        ----------
        r: float
            Distance from origin to scalp electrode location in units of [µm]
        theta: float
            Polar angle between scalp electrode location and
            dipole location vector rzloc in units of (radians)

        Returns
        -------
        pot_sum: float
            Summation factor for calculation of electrical potential in scalp
            from radial current dipole moment. (unitless)
        """
        n = 1
        const = 1.
        coeff_sum = 0.
        consts = []
        while const > self.iteration_stop_factor * coeff_sum:
            c4n = self._calc_c4n(n)
            d4n = self._calc_d4n(n, c4n)
            const = n * (c4n * (r / self.r4) ** n +
                         d4n * (self.r4 / r) ** (n + 1))
            coeff_sum += const
            consts.append(const)
            n += 1
        # since the legendre function starts with P0
        consts = np.insert(consts, 0, 0)
        leg_consts = np.polynomial.legendre.Legendre(consts)
        pot_sum = leg_consts(np.cos(theta))
        return pot_sum

    def _potential_brain_tan(self, r, theta):
        """
        Return factor for calculation of potential in brain from tan. dipole

        Parameters
        ----------
        r: float
            Distance from origin to brain electrode location in units of [µm]
        theta: float
            Polar angle between brain electrode location and
            dipole location vector rzloc in units of (radians)

        Returns
        -------
        pot_sum: float
            Summation factor for calculation of electrical potential in brain
            from tangential current dipole moment. (unitless)
        """
        n = 1
        const = 1.
        coeff_sum = 0.
        consts = []
        while const > self.iteration_stop_factor * coeff_sum:
            c1n = self._calc_c1n(n)
            const = (c1n * (r / self.r1) ** n + (self._rz / r) ** (n + 1))
            coeff_sum += const
            consts.append(const)
            n += 1
        pot_sum = np.sum([c * lpmv(1, i, np.cos(theta))
                          for c, i in zip(consts, np.arange(1, n))])
        return pot_sum

    def _potential_csf_tan(self, r, theta):
        """
        Return factor for calculation of potential in CSF from tan. dipole

        Parameters
        ----------
        r: float
            Distance from origin to CSF electrode location in units of [µm]
        theta: float
            Polar angle between CSF electrode location and
            dipole location vector rzloc in units of (radians)

        Returns
        -------
        pot_sum: float
            Summation factor for calculation of electrical potential in CSF
            from tangential current dipole moment. (unitless)
        """
        n = 1
        const = 1.
        coeff_sum = 0.
        consts = []
        while const > self.iteration_stop_factor * coeff_sum:
            term1 = self._calc_csf_term1(n, r)
            term2 = self._calc_csf_term2(n, r)
            const = term1 + term2
            coeff_sum += const
            consts.append(const)
            n += 1
        pot_sum = np.sum([c * lpmv(1, i, np.cos(theta))
                          for c, i in zip(consts, np.arange(1, n))])
        return pot_sum

    def _potential_skull_tan(self, r, theta):
        """
        Return factor for calculation of potential in skull from tan. dipole

        Parameters
        ----------
        r: float
            Distance from origin to skull electrode location in units of [µm]
        theta: float
            Polar angle between skull electrode location and
            dipole location vector rzloc in units of (radians)

        Returns
        -------
        pot_sum: float
            Summation factor for calculation of electrical potential in skull
            from tangential current dipole moment. (unitless)
        """
        n = 1
        const = 1.
        coeff_sum = 0.
        consts = []
        while const > self.iteration_stop_factor * coeff_sum:
            c3n = self._calc_c3n(n)
            d3n = self._calc_d3n(n, c3n)
            const = c3n * (r / self.r3) ** n + d3n * (self.r3 / r) ** (n + 1)
            coeff_sum += const
            consts.append(const)
            n += 1
        pot_sum = np.sum([c * lpmv(1, i, np.cos(theta))
                          for c, i in zip(consts, np.arange(1, n))])
        return pot_sum

    def _potential_scalp_tan(self, r, theta):
        """
        Return factor for calculation of potential in scalp from tan. dipole

        Parameters
        ----------
        r: float
            Distance from origin to scalp electrode location in units of [µm]
        theta: float
            Polar angle between scalp electrode location and
            dipole location vector rzloc in units of (radians)

        Returns
        -------
        pot_sum: float
            Summation factor for calculation of electrical potential in scalp
            from tangential current dipole moment. (unitless)
        """
        n = 1
        const = 1.
        coeff_sum = 0.
        consts = []
        while const > self.iteration_stop_factor * coeff_sum:
            c4n = self._calc_c4n(n)
            d4n = self._calc_d4n(n, c4n)
            const = c4n * (r / self.r4) ** n + d4n * (self.r4 / r) ** (n + 1)
            coeff_sum += const
            consts.append(const)
            n += 1
        pot_sum = np.sum([c * lpmv(1, i, np.cos(theta))
                          for c, i in zip(consts, np.arange(1, n))])
        return pot_sum

    def _calc_vn(self, n):
        r_const = ((self.r34 ** (2 * n + 1) - 1) /
                   ((n + 1) / n * self.r34 ** (2 * n + 1) + 1))
        if self.sigma23 + r_const == 0.0:
            v = 1e12
        else:
            v = (n / (n + 1) * self.sigma34 - r_const) / \
                (self.sigma34 + r_const)
        return v

    def _calc_yn(self, n):
        vn = self._calc_vn(n)
        r_const = ((n / (n + 1) * self.r23 ** (2 * n + 1) - vn) /
                   (self.r23 ** (2 * n + 1) + vn))
        if self.sigma23 + r_const == 0.0:
            y = 1e12
        else:
            y = (n / (n + 1) * self.sigma23 - r_const) / \
                (self.sigma23 + r_const)
        return y

    def _calc_zn(self, n):
        yn = self._calc_yn(n)
        z = (self.r12 ** (2 * n + 1) - (n + 1) / n * yn) / \
            (self.r12 ** (2 * n + 1) + yn)
        return z

    def _calc_c1n(self, n):
        zn = self._calc_zn(n)
        c1 = (((n + 1) / n * self.sigma12 + zn) /
              (self.sigma12 - zn) * self._rz1**(n + 1))
        return c1

    def _calc_c2n(self, n):
        yn = self._calc_yn(n)
        c1 = self._calc_c1n(n)
        c2 = ((c1 + self._rz1**(n + 1)) * self.r12 ** (n + 1) /
              (self.r12 ** (2 * n + 1) + yn))
        return c2

    def _calc_d2n(self, n, c2):
        yn = self._calc_yn(n)
        d2 = yn * c2
        return d2

    def _calc_c3n(self, n):
        vn = self._calc_vn(n)
        c2 = self._calc_c2n(n)
        d2 = self._calc_d2n(n, c2)
        c3 = (c2 + d2) * self.r23 ** (n + 1) / (self.r23 ** (2 * n + 1) + vn)
        return c3

    def _calc_d3n(self, n, c3):
        vn = self._calc_vn(n)
        d3 = vn * c3
        return d3

    def _calc_c4n(self, n):
        c3 = self._calc_c3n(n)
        d3 = self._calc_d3n(n, c3)
        c4 = ((n + 1) / n * self.r34 ** (n + 1) * (c3 + d3) /
              ((n + 1) / n * self.r34 ** (2 * n + 1) + 1))
        return c4

    def _calc_d4n(self, n, c4):
        d4 = n / (n + 1) * c4
        return d4

    def _calc_csf_term1(self, n, r):
        yn = self._calc_yn(n)
        c1 = self._calc_c1n(n)
        term1 = ((c1 + self._rz1 ** (n + 1)) * self.r12 * ((self.r1 * r) /
                 (self.r2 ** 2)) ** n / (self.r12**(2 * n + 1) + yn))
        return term1

    def _calc_csf_term2(self, n, r):
        yn = self._calc_yn(n)
        c1 = self._calc_c1n(n)
        term2 = (yn * (c1 + self._rz1 ** (n + 1)) /
                 (r / self.r2 * ((self.r1 * r) / self.r2**2) ** n +
                  (r / self.r1) ** (n + 1) * yn))
        return term2


class InfiniteVolumeConductor(object):
    """
    Main class for computing extracellular potentials with current dipole
    moment :math:`\\mathbf{P}` in an infinite 3D volume conductor model that
    assumes homogeneous, isotropic, linear (frequency independent)
    conductivity :math:`\\sigma`. The potential :math:`V` is computed as [1]_:

    .. math:: V = \\frac{\\mathbf{P} \\cdot \\mathbf{r}}{4 \\pi \\sigma r^3}

    Parameters
    ----------
    sigma: float
        Electrical conductivity in extracellular space in units of (S/cm)

    See also
    --------
    FourSphereVolumeConductor
    MEG

    References
    ----------
    .. [1] Nunez and Srinivasan, Oxford University Press, 2006

    Examples
    --------
    Computing the potential from dipole moment valid in the far field limit.

    >>> from lfpykit.eegmegcalc import InfiniteVolumeConductor
    >>> import numpy as np
    >>> inf_model = InfiniteVolumeConductor(sigma=0.3)
    >>> p = np.array([[10.], [10.], [10.]])  # [nA µm]
    >>> r = np.array([[1000., 0., 5000.]])  # [µm]
    >>> inf_model.get_dipole_potential(p, r)  # [mV]
    array([[1.20049432e-07]])
    """

    def __init__(self, sigma=0.3):
        "Initialize class InfiniteVolumeConductor"
        self.sigma = sigma

    def get_dipole_potential(self, p, r):
        """
        Return electric potential from current dipole moment ``p`` in
        locations ``r`` relative to dipole

        Parameters
        ----------
        p: ndarray, dtype=float
            Shape (3, n_timesteps) array containing the x,y,z components of the
            current dipole moment in units of (nA*µm) for all timesteps
        r: ndarray, dtype=float
            Shape (n_contacts, 3) array containing the displacement vectors
            from dipole location to measurement location

        Returns
        -------
        potential: ndarray, dtype=float
            Shape (n_contacts, n_timesteps) array containing the electric
            potential at contact point(s) ``r`` in units
            of [mV] for all timesteps of current dipole moment p

        """
        phi = (r @ p) / (4 * np.pi * self.sigma *
                         np.linalg.norm(r, axis=-1, keepdims=True)**3)
        return phi

    def get_transformation_matrix(self, r):
        '''
        Get linear response matrix mapping current dipole moment in [nA µm]
        to extracellular potential in [mV] at recording sites `r` [µm]

        parameters
        ----------
        r: ndarray, dtype=float
            Shape (n_contacts, 3) array contaning the displacement vectors
            from dipole location to measurement location [µm]

        Returns
        -------
        response_matrix: ndarray
            shape (n_contacts, 3) ndarray
        '''
        return self.get_dipole_potential(np.eye(3), r)


class MEG(object):
    """
    Basic class for computing magnetic field from current dipole moment.
    For this purpose we use the Biot-Savart law derived from Maxwell's
    equations under the assumption of negligible magnetic induction
    effects [1]_:

    .. math:: \\mathbf{H} = \\frac{\\mathbf{p} \\times \\mathbf{R}}{4 \\pi R^3}

    where :math:`\\mathbf{p}` is the current dipole moment, :math:`\\mathbf{R}`
    the vector between dipole source location and measurement location, and
    :math:`R=|\\mathbf{R}|`

    Note that the magnetic field :math:`\\mathbf{H}` is related to the magnetic
    field :math:`\\mathbf{B}` as

    .. math:: \\mu_0 \\mathbf{H} = \\mathbf{B}-\\mathbf{M}

    where :math:`\\mu_0` is the permeability of free space (very close to
    permebility of biological tissues). :math:`\\mathbf{M}` denotes material
    magnetization (also ignored)

    Parameters
    ----------
    sensor_locations: ndarray, dtype=float
        shape (n_locations x 3) array with x,y,z-locations of measurement
        devices where magnetic field of current dipole moments is calculated.
        In unit of [µm]
    mu: float
        Permeability. Default is permeability of vacuum
        (:math:`\\mu_0 = 4*\\pi*10^{-7}` T*m/A)

    See also
    --------
    FourSphereVolumeConductor
    InfiniteVolumeConductor

    References
    ----------
    .. [1] Nunez and Srinivasan, Oxford University Press, 2006

    Examples
    --------
    Define cell object, create synapse, compute current dipole moment:

    >>> import LFPy, os, numpy as np, matplotlib.pyplot as plt
    >>> from lfpykit.eegmegcalc import MEG
    >>> # create LFPy.Cell object
    >>> cell = LFPy.Cell(morphology=os.path.join(LFPy.__path__[0], 'test',
    >>>                                          'ball_and_sticks.hoc'),
    >>>                  passive=True)
    >>> cell.set_pos(0., 0., 0.)
    >>> # create single synaptic stimuli at soma (idx=0)
    >>> syn = LFPy.Synapse(cell, idx=0, syntype='ExpSyn', weight=0.01, tau=5,
    >>>                    record_current=True)
    >>> syn.set_spike_times_w_netstim()
    >>> # simulate, record current dipole moment
    >>> cell.simulate(rec_current_dipole_moment=True)
    >>> # Compute the dipole location as an average of segment locations
    >>> # weighted by membrane area:
    >>> dipole_location = (cell.area * np.c_[cell.xmid, cell.ymid, cell.zmid].T
    >>>                    / cell.area.sum()).sum(axis=1)
    >>> # Define sensor site, instantiate MEG object, get transformation matrix
    >>> sensor_locations = np.array([[1E4, 0, 0]])
    >>> meg = MEG(sensor_locations)
    >>> M = meg.get_transformation_matrix(dipole_location)
    >>> # compute the magnetic signal in a single sensor location:
    >>> H = M @ cell.current_dipole_moment.T
    >>> # plot output
    >>> plt.figure(figsize=(12, 8), dpi=120)
    >>> plt.subplot(311)
    >>> plt.plot(cell.tvec, cell.somav)
    >>> plt.ylabel(r'$V_{soma}$ (mV)')
    >>> plt.subplot(312)
    >>> plt.plot(cell.tvec, syn.i)
    >>> plt.ylabel(r'$I_{syn}$ (nA)')
    >>> plt.subplot(313)
    >>> plt.plot(cell.tvec, H[0].T)
    >>> plt.ylabel(r'$H$ (nA/um)')
    >>> plt.xlabel('$t$ (ms)')
    >>> plt.legend(['$H_x$', '$H_y$', '$H_z$'])
    >>> plt.show()

    Raises
    ------
    AssertionError
        If dimensionality of sensor_locations is wrong
    """
    def __init__(self, sensor_locations, mu=4 * np.pi * 1E-7):
        """
        Initialize class MEG
        """
        assert sensor_locations.ndim == 2, 'sensor_locations.ndim != 2'
        assert sensor_locations.shape[1] == 3, 'sensor_locations.shape[1] != 3'

        # set attributes
        self.sensor_locations = sensor_locations
        self.mu = mu

    def get_transformation_matrix(self, dipole_location):
        '''
        Get linear response matrix mapping current dipole moment in [nA µm]
        located in location ``dipole_location` to magnetic field
        :math:`\\mathbf{H}` in units of (nA/µm) at ``sensor_locations``

        parameters
        ----------
        dipole_location: ndarray, dtype=float
            shape (3, ) array with x,y,z-location of dipole in units of [µm]

        Returns
        -------
        response_matrix: ndarray
            shape (n_contacts, 3, 3) ndarray
        '''
        return self.calculate_H(np.eye(3), dipole_location)

    def calculate_H(self, current_dipole_moment, dipole_location):
        """
        Compute magnetic field H from single current-dipole moment localized
        somewhere in space

        Parameters
        ----------
        current_dipole_moment: ndarray, dtype=float
            shape (3, n_timesteps) array with x,y,z-components of current-
            dipole moment time series data in units of (nA µm)
        dipole_location: ndarray, dtype=float
            shape (3, ) array with x,y,z-location of dipole in units of [µm]

        Returns
        -------
        ndarray, dtype=float
            shape (n_locations x 3 x n_timesteps) array with x,y,z-components
            of the magnetic field :math:`\\mathbf{H}` in units of (nA/µm)

        Raises
        ------
        AssertionError
            If dimensionality of current_dipole_moment and/or dipole_location
            is wrong
        """
        assert current_dipole_moment.ndim == 2, \
            'current_dipole_moment.ndim != 2'
        assert current_dipole_moment.shape[0] == 3, \
            'current_dipole_moment.shape[0] != 3'
        assert dipole_location.shape == (3, ), \
            'dipole_location.shape != (3, )'

        # container
        H = np.empty((self.sensor_locations.shape[0], 3,
                      current_dipole_moment.shape[1]))
        # iterate over sensor locations
        for i, r in enumerate(self.sensor_locations):
            R = r - dipole_location
            assert R.ndim == 1 and R.size == 3
            assert not np.allclose(R, np.zeros(3)), \
                'Identical dipole and sensor location.'

            H[i, ] = np.cross(current_dipole_moment.T, R).T \
                / (4 * np.pi * np.sqrt((R**2).sum())**3)

        return H


class NYHeadModel(object):
    """
    Main class for computing EEG signals from current dipole
    moment :math:`\\mathbf{P}` in New York Head Model [1]_, [2]_

    Assumes units of nA * um for current dipole moment, and mV for the EEG

    Notes
    -----
    The original unit of the New York model current dipole moment
    is (probably?) mA*m, and the EEG output unit is V.
    LFPykit's current dipole moments have units nA*um, and EEGs from the
    NYhead model is here recomputed in units of mV.

    Parameters
    ----------
    nyhead_file: str [optional]
        Location of file containing New York Head Model. If empty (or None),
        it will be looked for in the main LFPykit folder. If not present
        the user is asked if it should be downloaded from
        https://www.parralab.org/nyhead/sa_nyhead.mat

    See also
    --------
    FourSphereVolumeConductor
    MEG

    References
    ----------
    .. [1] Huang, Parra, Haufe (2016) The New York Head—A precise standardized
        volume conductor model for EEG source localization and tES targeting.
        Neuroimage 140:150–162. doi: 10.1016/j.neuroimage.2015.12.019
    .. [2] Naess et al. (2020) Biophysical modeling of the neural origin of EEG
        and MEG signals. bioRxiv 2020.07.01.181875.
        doi: 10.1101/2020.07.01.181875

    Examples
    --------
    Computing EEG from dipole moment.

    >>> from lfpykit.eegmegcalc import NYHeadModel

    >>> nyhead = NYHeadModel()

    >>> nyhead.set_dipole_pos('parietal_lobe') # predefined example location
    >>> M = nyhead.get_transformation_matrix()

    >>> # Rotate to be along normal vector of cortex
    >>> p = nyhead.rotate_dipole_to_surface_normal([[0.], [0.], [1.]])
    >>> eeg = M @ p  # [mV]
    """

    def __init__(self, nyhead_file=None):
        """ Initialize class NYHeadModel """

        # Some example locations in NY Head model
        self.dipole_pos_dict = {
            'calcarine_sulcus': np.array([5, -86, 2]),
            'motorsensory_cortex': np.array([18, 8, 71]),
            'parietal_lobe': np.array([54, -48, 55.9]),
            'occipital_lobe': np.array([-24.7, -103.3, -1.46])
        }

        self._load_head_model(nyhead_file)

        # These will be set by the "set_dipole_pos" function:
        self.dipole_pos = None
        self.cortex_normal_vec = None
        self.vertex_idx = None

    def _load_head_model(self, nyhead_file):
        try:
            import h5py
        except ModuleNotFoundError:
            raise ImportError("The package h5py was not found. "
                              "It is needed for loading New York Head model.")

        if nyhead_file is None:
            lfpykit_folder = os.path.dirname(os.path.realpath(__file__))
            nyhead_file = os.path.join(lfpykit_folder, "sa_nyhead.mat")
        self.head_file = os.path.abspath(nyhead_file)
        if not os.path.isfile(self.head_file):
            from urllib.request import urlopen
            import ssl
            print("New York head model not found: %s" % self.head_file)
            yn = input("Should it be downloaded (710 MB)? [y/n]: ")
            if yn == 'y':
                print("Now downloading. This might take a while ...")
                nyhead_url = 'https://www.parralab.org/nyhead/sa_nyhead.mat'
                u = urlopen(nyhead_url,
                            context=ssl._create_unverified_context())
                localFile = open(self.head_file, 'wb')
                localFile.write(u.read())
                localFile.close()
                print("Download done!")
            else:
                print("Exiting program ...")
                sys.exit()

        self.head_data = h5py.File(self.head_file, 'r')["sa"]
        self.cortex = np.array(self.head_data["cortex75K"]["vc"])
        self.lead_field = np.array(self.head_data["cortex75K"]["V_fem"])
        self.lead_field_normal = np.array(
            self.head_data["cortex75K"]["V_fem_normal"])
        self.cortex_normals = np.array(self.head_data["cortex75K"]["normals"])
        self.elecs = np.array(self.head_data["locs_3D"])

        # Other properties that might be used for plotting
        # self.sulicmap = np.array(f["sa"]["cortex75K"]["sulcimap"])[0,:]
        # self.head = np.array(self.head_data["head"]["vc"])
        # self.head_tri = np.array(self.head_data["head"]["tri"],
        #                          dtype=int) - 1
        # self.cortex_tri = np.array(
        #     self.head_data["cortex75K"]["tri"], dtype=int)[:, :] - 1

    def rotate_dipole_to_surface_normal(self, p, orig_ax_vec=[0, 0, 1]):
        """
        Returns rotated dipole moment, p_rot, oriented along the normal
        vector of the cortex at the dipole location

        Parameters
        ----------
        p : np.ndarray of size (3, num_timesteps)
            Current dipole moment from neural simulation,
            [p_x(t), p_y(t), p_z(t)]. If z-axis is the depth axis of cortex
            in the original neural simulation p_x(t) and p_y(t) will
            typically be small, and orig_ax_vec = [0, 0, 1].
        orig_ax_vec : np.ndarray or list of length (3)
            Original surface vector of cortex in the neural simulation. If
            depth axis of cortex is the z-axis, orig_ax_vec = [0, 0, 1].

        Returns
        -------
        p_rot : np.ndarray of size (3, num_timesteps)
            Rotated current dipole moment, oriented along cortex normal vector
            at the dipole location

        References
        ----------
        See: https://en.wikipedia.org/wiki/Rotation_matrix
        under "Rotation matrix from axis and angle"
        """

        if self.cortex_normal_vec is None:
            raise RuntimeError("Dipole location must first be set by " +
                               "set_dipole_pos(loc) function.")
        surface_vec = self.cortex_normal_vec

        surface_vec /= np.linalg.norm(surface_vec)
        orig_ax_vec /= np.linalg.norm(orig_ax_vec)

        # rotation angle
        phi = math.acos(np.dot(orig_ax_vec, surface_vec))

        # axis to rotate around
        rot_axis = np.cross(orig_ax_vec, surface_vec)
        axis_len = np.linalg.norm(rot_axis)
        if axis_len > 1e-9:
            rot_axis /= axis_len
        x_, y_, z_ = rot_axis

        # calculate rotation matrix
        cos_th = np.cos(phi)
        sin_th = np.sin(phi)
        R = np.zeros((3, 3))
        R[0, 1] = -z_ * sin_th + (1.0 - cos_th) * x_ * y_
        R[0, 2] = +y_ * sin_th + (1.0 - cos_th) * x_ * z_
        R[1, 0] = +z_ * sin_th + (1.0 - cos_th) * x_ * y_
        R[1, 2] = -x_ * sin_th + (1.0 - cos_th) * y_ * z_
        R[2, 0] = -y_ * sin_th + (1.0 - cos_th) * x_ * z_
        R[2, 1] = +x_ * sin_th + (1.0 - cos_th) * y_ * z_
        R[0, 0] = cos_th + x_**2 * (1 - cos_th)
        R[1, 1] = cos_th + y_**2 * (1 - cos_th)
        R[2, 2] = cos_th + z_**2 * (1 - cos_th)

        return R @ p

    def return_closest_idx(self, pos):
        """
        Returns the index of the closest vertex in the brain to a given
        position (in mm).

        Parameters
        ----------
        pos : array of length (3)
            [x, y, z] of a location in the brain, given in mm, and not in um
            which is the default position unit in LFPy
        Returns
        -------
        idx : int
            Index of the vertex in the brain that is closest to the given
            location
        """
        return np.argmin((self.cortex[0, :] - pos[0])**2 +
                         (self.cortex[1, :] - pos[1])**2 +
                         (self.cortex[2, :] - pos[2])**2)

    def find_closest_electrode(self):
        """
        Returns minimal distance (mm) and closest electrode idx to
        dipole location specified in self.dipole_pos.

        """
        if self.dipole_pos is None:
            raise RuntimeError("Dipole location must first be set by " +
                               "set_dipole_pos(loc) function.")

        dists = (np.sqrt(np.sum((np.array(self.dipole_pos)[:, None] -
                                 np.array(self.elecs[:3, :]))**2, axis=0)))
        closest_electrode = np.argmin(dists)
        min_dist = np.min(dists)
        return min_dist, closest_electrode

    def set_dipole_pos(self, dipole_pos=None):
        """
        Sets the dipole location in the brain

        Parameters
        ----------
        dipole_pos: None, str or array of length (3) [x, y, z) (mm)
            Location of the dipole. If no argument is given
            (or dipole_pos=None), a location, 'motorsensory_cortex',
            from self.dipole_pos_dict is used. If dipole_pos is an
            array of length 3, the closest vertex in the brain will be
            set as the dipole location.

        """
        if dipole_pos is None:
            dipole_pos_ = self.dipole_pos_dict['motorsensory_cortex']
        elif type(dipole_pos) is str:
            if dipole_pos not in self.dipole_pos_dict:
                raise RuntimeError("When dipole_pos is string, location must"
                                   "be defined in self.dipole_pos_dict. "
                                   "Choose one of: {}".format(
                                        self.dipole_pos_dict.keys()))
            dipole_pos_ = self.dipole_pos_dict[dipole_pos]
        elif type(dipole_pos) not in [list, np.ndarray]:
            raise RuntimeError("dipole_pos argument type is not valid. "
                               "Must be None, str, or array")
        elif type(dipole_pos) in [list, np.ndarray] and len(dipole_pos) != 3:
            raise RuntimeError("If dipole_pos argument is array it must "
                               "have length 3")
        else:
            dipole_pos_ = dipole_pos

        self.vertex_idx = self.return_closest_idx(dipole_pos_)
        self.dipole_pos = self.cortex[:, self.vertex_idx]
        loc_error = np.sqrt(np.sum((self.dipole_pos - dipole_pos_)**2))

        if loc_error > 2:
            raise RuntimeWarning("Large dipole location error! "
                                 "Given loc: {}; Closest vertex: {}".format(
                                    dipole_pos_, self.dipole_pos))

        self.cortex_normal_vec = self.cortex_normals[:, self.vertex_idx]

    def get_transformation_matrix(self):
        """
        Get linear response matrix mapping from current dipole moment [nA µm]
        to EEG signal [mV] at EEG electrodes (n=231)

        Returns
        -------
        response_matrix: ndarray
            shape (231, 3) ndarray
        """
        return self.lead_field[:, self.vertex_idx, :].T * 1E-9
