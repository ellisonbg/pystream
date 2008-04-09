# -*- coding: utf-8 -*-
"""ctypes wrapper for cublas."""

#----------------------------------------------------------------------------
# Copyright (c) 2007, Tech-X Corporation
#----------------------------------------------------------------------------



#----------------------------------------------------------------------------
# Imports
#----------------------------------------------------------------------------

import platform
from ctypes import *

if platform.system() == "Microsoft":
    libcublas = ctypes.windll.LoadLibrary('cublas.dll')
elif platform.system()=="Darwin":
    libcublas = ctypes.cdll.LoadLibrary('/usr/local/cuda/lib/libcublas.dylib')
else:
    libcublas = ctypes.cdll.LoadLibrary('libcublas.so')

# defines

CUBLAS_STATUS_SUCCESS           = 0x00000000
CUBLAS_STATUS_NOT_INITIALIZED   = 0x00000001
CUBLAS_STATUS_ALLOC_FAILED      = 0x00000003
CUBLAS_STATUS_INVALID_VALUE     = 0x00000007
CUBLAS_STATUS_MAPPING_ERROR     = 0x0000000B
CUBLAS_STATUS_EXECUTION_FAILED  = 0x0000000D
CUBLAS_STATUS_INTERNAL_ERROR    = 0x0000000E

cublasStatus = c_uint

# Exceptions

class CublasError(Exception):
    pass

def checkCublasStatus(status):
    if status != CUBLAS_STATUS_SUCCESS:
        raise CublasError("Internal cuda error: %i" % status)

# Helper functions

# cublasInit
_cublasInit = libcublas.cublasInit
_cublasInit.restype = cublasStatus
_cublasInit.argtypes = []

def cublasInit():
    status = _cublasInit()
    checkCublasStatus(status)

# cublasShutdown
_cublasShutdown = libcublas.cublasShutdown
_cublasShutdown.restype = cublasStatus
_cublasShutdown.argtypes = []

def cublasShutdown():
    status = _cublasShutdown()
    checkCublasStatus(status)    

# cublasGetError
_cublasGetError = libcublas.cublasGetError
_cublasGetError.restype = cublasStatus
_cublasGetError.argtypes = []

def cublasGetError():
    status = _cublasGetError()
    checkCublasStatus(status)

# cublasAlloc
_cublasAlloc = libcublas.cublasAlloc
_cublasAlloc.restype = cublasStatus
_cublasAlloc.argtypes = [c_int, c_int, POINTER(c_void_p)]

def cublasAlloc(n, ctype):
    assert isinstance(n, int)
    devPtr = pointer(ctype())
    castDevPtr = cast(pointer(devPtr), POINTER(c_void_p))
    status = _cublasAlloc(n, sizeof(ctype), castDevPtr)
    checkCublasStatus(status)
    return devPtr

# cublasFree
_cublasFree = libcublas.cublasFree
_cublasFree.restype = cublasStatus
_cublasFree.argtypes = [c_void_p]

def cublasFree(devPtr):
    status = _cublasFree(devPtr)
    checkCublasStatus(status)

# cublasSetVector
_cublasSetVector = libcublas.cublasSetVector
_cublasSetVector.restype = cublasStatus
_cublasSetVector.argtypes = [c_int, c_int, c_void_p, c_int,
                             c_void_p, c_int]

def cublasSetVector(n, ctype, hostVector, incx, deviceVector, incy):
    assert isinstance(n, int)
    assert isinstance(incx, int)
    assert isinstance(incy, int)
    elemSize = sizeof(ctype)
    status = _cublasSetVector(n, elemSize, hostVector, incx,
        deviceVector, incy)
    checkCublasStatus(status)    
    

# cublasGetVector
_cublasGetVector = libcublas.cublasGetVector
_cublasGetVector.restype = cublasStatus
_cublasGetVector.argtypes = [c_int, c_int, c_void_p, c_int,
                             c_void_p, c_int]

def cublasGetVector(n, ctype, deviceVector, incx, hostVector, incy):
    assert isinstance(n, int)
    assert isinstance(incx, int)
    assert isinstance(incy, int)
    elemSize = sizeof(ctype)
    status = _cublasGetVector(n, elemSize, deviceVector, incx,
        hostVector, incy)
    checkCublasStatus(status)   

# cublasSetMatrix
_cublasSetMatrix = libcublas.cublasSetMatrix
_cublasSetMatrix.restype = cublasStatus
_cublasSetMatrix.argtypes = [c_int, c_int, c_int, c_void_p, c_int,
                             c_void_p, c_int]

def cublasSetMatrix(rows, cols, ctype, hostMatrixA, lda, 
                    deviceMatrixB, ldb):
    assert isinstance(rows, int)
    assert isinstance(cols, int)
    assert isinstance(lda, int)
    assert isinstance(ldb, int)
    elemSize = sizeof(ctype)
    status = _cublasSetMatrix(rows, cols, elemSize, hostMatrixA, lda, 
                    deviceMatrixB, ldb)
    checkCublasStatus(status) 

# cublasGetMatrix
_cublasGetMatrix = libcublas.cublasGetMatrix
_cublasGetMatrix.restype = cublasStatus
_cublasGetMatrix.argtypes = [c_int, c_int, c_int, c_void_p, c_int,
                             c_void_p, c_int]

def cublasGetMatrix(rows, cols, ctype, deviceMatrixA, lda, 
                    hostMatrixB, ldb):
    assert isinstance(rows, int)
    assert isinstance(cols, int)
    assert isinstance(lda, int)
    assert isinstance(ldb, int)
    elemSize = sizeof(ctype)
    status = _cublasGetMatrix(rows, cols, elemSize, deviceMatrixA, lda, 
                    hostMatrixB, ldb)
    checkCublasStatus(status)

# BLAS1


# BLAS2

# BLAS3

_cublasSgemm = libcublas.cublasSgemm
_cublasSgemm.restype = None
_cublasSgemm.argtypes = [c_char, c_char, c_int, c_int, c_int,
                         c_float, POINTER(c_float), c_int,
                         POINTER(c_float), c_int, c_float, 
                         POINTER(c_float), c_int]

def cublasSgemm(transa, transb, m, n, k, alpha, A,
                lda, B, ldb, beta, C, ldc):
    _cublasSgemm(transa, transb, m, n, k, alpha, A, lda,
                          B, ldb, beta, C, ldc)
    cublasGetError()





