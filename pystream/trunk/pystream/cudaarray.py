# -*- coding: utf-8 -*-
"""Array-like objects for CUDA."""

#----------------------------------------------------------------------------
# Copyright (c) 2007, Tech-X Corporation
#----------------------------------------------------------------------------



#----------------------------------------------------------------------------
# Imports
#----------------------------------------------------------------------------

import cudart
import numpy
import ctypes

# cuda <-> dtype conversion
cudaDtypes = {'float32': ctypes.c_float,
              'int32': ctypes.c_int}

class CudaArray(object):

    def __init__(self, size, dtype):
        self.size = size
        self.dtype = numpy.dtype(dtype)
        self.ctype = self._convertType(self.dtype)
        self.nbytes = self.size*ctypes.sizeof(self.ctype)
        self.allocated = False
        self.data = None

    def __del__(self):
        self.free()

    def _convertType(self, dtype):
        ct = cudaDtypes.get(dtype.name, None)
        if ct is None:
            raise Exception("Unsupported dtype")
        return ct

    def alloc(self):
        self.data = cudart.cudaMalloc(self.nbytes, self.ctype)
        self.allocated = True

    def free(self):
        if self.allocated:
            cudart.cudaFree(self.data)
            self.data = None
            self.allocated = False

    def toArray(self):
        if not self.allocated:
            raise Exception("Must first allocate")
        a = numpy.empty(self.size, dtype=self.dtype)
        cudart.cudaMemcpy(a.ctypes.data, self.data, self.nbytes,
            cudart.cudaMemcpyDeviceToHost)
        return a

    def setWithArray(self, a):
        if not self.allocated:
            raise Exception("Must first allocate")
        a = numpy.ascontiguousarray(a, dtype=None)
        assert a.size == self.size, "size must be the same"
        assert a.dtype == self.dtype, "dtype must be the same"
        cudart.cudaMemcpy(self.data, a.ctypes.data, self.nbytes,
            cudart.cudaMemcpyHostToDevice)

class RawCudaArray(CudaArray):

    def __init__(self, size, dtype):
        CudaArray.__init__(self, size, dtype)
        self.alloc()

class CudaArrayFromArray(CudaArray):
    def __init__(self, a, dtype=None):
        a = numpy.ascontiguousarray(a, dtype=dtype)
        CudaArray.__init__(self, a.size, a.dtype)
        self.alloc()
        self.setWithArray(a)

