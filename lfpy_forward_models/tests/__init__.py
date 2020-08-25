#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Copyright (C) 2012 Computational Neuroscience Group, NMBU.
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
"""


def _test(verbosity=1):
    """run all unit tests

    Parameters
    ----------
    verbosity : int
        unittest.TestCase verbosity level, default is 1

    """
    # import methods here to avoid polluting namespace
    from test_module import TestSuite
    from test_lfpcalc import testLfpCalc
    import unittest

    print('\ntest lfpy_forward_models main classes:')
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSuite)
    unittest.TextTestRunner(verbosity=verbosity).run(suite)

    print('\ntest lfpy_forward_models.lfpcalc methods:')
    suite = unittest.TestLoader().loadTestsFromTestCase(testLfpCalc)
    unittest.TextTestRunner(verbosity=verbosity).run(suite)


if __name__ == '__main__':
    _test()
