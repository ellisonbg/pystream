# -*- coding: utf-8 -*-
"""ctypes wrapper for cufft."""

#----------------------------------------------------------------------------
# Copyright (c) 2007, Enthought, Inc.
#----------------------------------------------------------------------------

# TODO
#   * Make a hierarchy of exceptions.
#   * Make simple fft(), ifft(), etc. functions.


#----------------------------------------------------------------------------
# Imports
#----------------------------------------------------------------------------

import ctypes
from ctypes import byref, c_int, POINTER


libcufft = ctypes.cdll.LoadLibrary('libcufft.so')

# Definitions

CUFFT_FORWARD = -1
CUFFT_INVERSE = 1

# cufftResult_t, cufftResult
CUFFT_SUCCESS        = 0
CUFFT_INVALID_PLAN   = 1
CUFFT_ALLOC_FAILED   = 2
CUFFT_INVALID_TYPE   = 3
CUFFT_INVALID_VALUE  = 4
CUFFT_INTERNAL_ERROR = 5
CUFFT_EXEC_FAILED    = 6
CUFFT_SETUP_FAILED   = 7
CUFFT_INVALID_SIZE   = 8
    
# cufftType_t, cufftType
CUFFT_R2C = 0x2a
CUFFT_C2R = 0x2c
CUFFT_C2C = 0x29

cufftType_enum = [CUFFT_R2C, CUFFT_C2R, CUFFT_C2C]

# Typedefs
cufftResult = ctypes.c_int
cufftType = ctypes.c_int
cufftHandle = ctypes.c_uint
cufftReal = ctypes.c_float
cufftComplex = cufftReal * 2


# Exceptions

class CufftError(Exception):
    pass

def checkCufftResult(result):
    if result != CUFFT_SUCCESS:
        raise CufftError("Internal cuda error: %i" % result)


_cufftPlan1d = libcufft.cufftPlan1d
_cufftPlan1d.restype = cufftResult
_cufftPlan1d.argtypes = [POINTER(cufftHandle), c_int, cufftType, c_int]

def cufftPlan1d(nx, type_, batch=1):
    """ Create a plan for doing a 1D FFT.

    Parameters
    ----------
    `nx` : int
        The number of elements in the array.
    `type_` : cufftType
        The kind of FFT (real->complex, complex->real, complex->complex).
    `batch` : int, optional
        The number of transforms of this size to batch up.

    Returns
    -------
    `handle` : cufftHandle
        An opaque handle to the plan.

    Raises
    ------
    AssertionError if the arguments are bad.
    CufftError if there is an error calling the CUFFT library.
    """
    assert isinstance(nx, int)
    assert 2 <= nx <= 8*1024*1024   # XXX: CUDA 1.0; may change
    assert type_ in cufftType_enum
    if batch is None:
        batch = 1
    assert isinstance(batch, int)
    assert batch >= 1
    plan = cufftHandle()
    result = _cufftPlan1d(byref(plan), nx, type_, batch)
    checkCufftResult(result)
    return plan

_cufftPlan2d = libcufft.cufftPlan2d
_cufftPlan2d.restype = cufftResult
_cufftPlan2d.argtypes = [POINTER(cufftHandle), c_int, c_int, cufftType]

def cufftPlan2d(nx, ny, type_):
    """ Create a plan for doing a 2D FFT.

    Parameters
    ----------
    `nx` : int
    `ny` : int
        The dimensions of the 2D array.
    `type_` : cufftType
        The kind of FFT (real->complex, complex->real, complex->complex).
    `batch` : int, optional
        The number of transforms of this size to batch up.

    Returns
    -------
    `handle` : cufftHandle
        An opaque handle to the plan.

    Raises
    ------
    AssertionError if the arguments are bad.
    CufftError if there is an error calling the CUFFT library.
    """
    assert isinstance(nx, int)
    assert 2 <= nx <= 16384   # XXX: CUDA 1.0; may change
    assert isinstance(ny, int)
    assert 2 <= ny <= 16384   # XXX: CUDA 1.0; may change
    assert type_ in cufftType_enum
    plan = cufftHandle()
    result = _cufftPlan2d(byref(plan), nx, ny, type_)
    checkCufftResult(result)
    return plan

_cufftPlan3d = libcufft.cufftPlan3d
_cufftPlan3d.restype = cufftResult
_cufftPlan3d.argtypes = [POINTER(cufftHandle), c_int, c_int, c_int, cufftType]

def cufftPlan3d(nx, ny, nz, type_):
    """ Create a plan for doing a 3D FFT.

    Parameters
    ----------
    `nx` : int
    `ny` : int
    `nz` : int
        The dimensions of the 3D array.
    `type_` : cufftType
        The kind of FFT (real->complex, complex->real, complex->complex).
    `batch` : int, optional
        The number of transforms of this size to batch up.

    Returns
    -------
    `handle` : cufftHandle
        An opaque handle to the plan.

    Raises
    ------
    AssertionError if the arguments are bad.
    CufftError if there is an error calling the CUFFT library.
    """
    assert isinstance(nx, int)
    assert 2 <= nx <= 16384   # XXX: CUDA 1.0; may change
    assert isinstance(ny, int)
    assert 2 <= ny <= 16384   # XXX: CUDA 1.0; may change
    assert isinstance(nz, int)
    assert 2 <= nz <= 16384   # XXX: CUDA 1.0; may change
    assert type_ in cufftType_enum
    plan = cufftHandle()
    result = _cufftPlan3d(byref(plan), nx, ny, nz, type_)
    checkCufftResult(result)
    return plan

_cufftDestroy = libcufft.cufftDestroy
_cufftDestroy.restype = cufftResult
_cufftDestroy.argtypes = [cufftHandle]

def cufftDestroy(plan):
    """ Destroy the given plan.

    Parameters
    ----------
    `plan` : cufftHandle
        The plan handle.

    Raises
    ------
    AssertionError if the argument is bad.
    CufftError if there is an error calling the CUFFT library.
    """
    assert isinstance(plan, cufftHandle)
    result = _cufftDestroy(plan)
    checkCufftResult(result)
    

_cufftExecC2C = libcufft.cufftExecC2C
_cufftExecC2C.restype = cufftResult
_cufftExecC2C.argtypes = [cufftHandle, POINTER(cufftComplex),
    POINTER(cufftComplex), c_int]

def cufftExecC2C(plan, idata, odata, direction=CUFFT_FORWARD):
    """ Execute the planned complex->complex FFT.

    Parameters
    ----------
    `plan` : cufftHandle
        The plan handle.
    `idata` : pointer to cufftComplex array
    `odata` : pointer to cufftComplex array
        The input and output arrays. They may be the same for an in-place FFT.
    `direction` : int, optional
        Either CUFFT_FORWARD or CUFFT_INVERSE.

    Raises
    ------
    AssertionError if the argument is bad.
    CufftError if there is an error calling the CUFFT library.
    """
    assert isinstance(plan, cufftHandle)
    # TODO: check pointer validity.
    # TODO: accept contiguous numpy arrays.
    assert direction in (CUFFT_FORWARD, CUFFT_INVERSE)
    result = _cufftExecC2C(plan, idata, odata, direction)
    checkCufftResult(result)

_cufftExecR2C = libcufft.cufftExecR2C
_cufftExecR2C.restype = cufftResult
_cufftExecR2C.argtypes = [cufftHandle, POINTER(cufftReal),
    POINTER(cufftComplex)]

def cufftExecR2C(plan, idata, odata):
    """ Execute the planned real->complex FFT.

    The transform is implicitly forward.

    Parameters
    ----------
    `plan` : cufftHandle
        The plan handle.
    `idata` : pointer to cufftReal array
    `odata` : pointer to cufftComplex array
        The input and output arrays.

    Raises
    ------
    AssertionError if the argument is bad.
    CufftError if there is an error calling the CUFFT library.
    """
    assert isinstance(plan, cufftHandle)
    # TODO: check pointer validity.
    # TODO: accept contiguous numpy arrays.
    result = _cufftExecR2C(plan, idata, odata)
    checkCufftResult(result)


_cufftExecC2R = libcufft.cufftExecC2R
_cufftExecC2R.restype = cufftResult
_cufftExecC2R.argtypes = [cufftHandle, POINTER(cufftComplex),
    POINTER(cufftComplex)]

def cufftExecC2R(plan, idata, odata):
    """ Execute the planned complex->real FFT.

    The transform is implicitly inverse.

    Parameters
    ----------
    `plan` : cufftHandle
        The plan handle.
    `idata` : pointer to cufftComplex array
    `odata` : pointer to cufftReal array
        The input and output arrays.

    Raises
    ------
    AssertionError if the argument is bad.
    CufftError if there is an error calling the CUFFT library.
    """
    assert isinstance(plan, cufftHandle)
    # TODO: check pointer validity.
    # TODO: accept contiguous numpy arrays.
    result = _cufftExecC2R(plan, idata, odata)
    checkCufftResult(result)


