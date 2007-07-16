# -*- coding: utf-8 -*-
"""basic tests for cudart."""

#----------------------------------------------------------------------------
# Copyright (c) 2007, Tech-X Corporation
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
# Imports
#----------------------------------------------------------------------------

import unittest
from ctypes import *
from pystream import cudart, cudaarray
import numpy as N

#----------------------------------------------------------------------------
# Testing utilities
#----------------------------------------------------------------------------




#----------------------------------------------------------------------------
# The TestCases
#----------------------------------------------------------------------------

class TestCudaArray(unittest.TestCase):

    def testRawCudaArray(self):
        for size in [2**k for k in range(10)]:
            for dt in ('int32', 'float32', 'complex64'):
                ca = cudaarray.RawCudaArray(size, dtype=dt)
                a = N.arange(size, dtype=dt)
                ca.setWithArray(a)
                b = ca.toArray()
                self.assert_(N.allclose(a,b))

    def testCudaArrayFromArray(self):
        for size in [2**k for k in range(10)]:
            for dt in ('int32', 'float32', 'complex64'):
                a = N.arange(size, dtype=dt)
                ca = cudaarray.CudaArrayFromArray(a, dtype=dt)
                b = ca.toArray()
                self.assert_(N.allclose(a, b))        



if __name__ == '__main__':
    try:
        unittest.main()
    except SystemExit:
        pass
