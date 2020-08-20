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


import os
import unittest
import numpy as np
import lfpy_forward_models

class TestSuite(unittest.TestCase):
    """
    test methods and modules
    """
    def test_testsuite_00(self):
        '''test TestSuite'''
        self.assertTrue(True)
