# -*- coding: utf-8 -*-
"""ctypes wrapper for cudart."""

#----------------------------------------------------------------------------
# Copyright (c) 2007, Tech-X Corporation
#----------------------------------------------------------------------------



#----------------------------------------------------------------------------
# Imports
#----------------------------------------------------------------------------

from ctypes import *

libcudart = cdll.LoadLibrary('libcudart.so')

#----------------------------------------------------------------------------
# Global constants and exceptions
#----------------------------------------------------------------------------


cudaSuccess = 0

errorDict = {
    1: 'MissingConfigurationError',
    2: 'MemoryAllocationError',
    3: 'InitializationError',
    4: 'LaunchFailureError',
    5: 'PriorLaunchFailureError',
    6: 'LaunchTimeoutError',
    7: 'LaunchOutOfResourcesError',
    8: 'InvalidDeviceFunctionError',
    9: 'InvalidConfigurationError',
    10: 'InvalidDeviceError',
    11: 'InvalidValueError',
    12: 'InvalidPitchValueError',
    13: 'InvalidSymbolError',
    14: 'MapBufferObjectFailedError',
    15: 'UnmapBufferObjectFailedError',
    16: 'InvalidHostPointerError',
    17: 'InvalidDevicePointerError',
    18: 'InvalidTextureError',
    19: 'InvalidTextureBindingError',
    20: 'InvalidChannelDescriptorError',
    21: 'InvalidMemcpyDirectionError',
    22: 'AddressOfConstantError',
    23: 'TextureFetchFailedError',
    24: 'TextureNotBoundError',
    25: 'SynchronizationError',
    26: 'InvalidFilterSettingError',
    27: 'InvalidNormSettingError',
    28: 'MixedDeviceExecutionError',
    29: 'CudartUnloadingError',
    30: 'UnknownError',
    31: 'NotYetImplementedError',
    0x7f: 'StartupFailureError',
    10000: 'ApiFailureBaseError'}

class CudaError(Exception):
    def __str__(self):
        a = '\n' + repr(self.args)
        return "Internal CUDA error: " + getErrorString(self.value) + a

for k, v in errorDict.iteritems():
    eString = """class %s(CudaError):
    value = %i""" % (v, k)
    exec eString in locals(), globals()

def _checkCudaStatus(status):
    if status != cudaSuccess:
        eClassString = errorDict[status]
        # Get the class by name from the top level of this module
        eClass = globals()[eClassString]
        raise eClass()

# enum cudaMemcpyKind
memcpyHostToHost = 0
memcpyHostToDevice = 1
memcpyDeviceToHost = 2
memcpyDeviceToDevice = 3

# cudaDeviceProp
class DeviceProp(Structure):
    _fields_ = [("name", 256*c_char),
                ("totalGlobalMem", c_size_t),
                ("sharedMemPerBlock", c_size_t),
                ("regsPerBlock", c_int),
                ("warpSize", c_int),
                ("memPitch", c_size_t),
                ("maxThreadsPerBlock", c_int),
                ("maxThreadsDim", 3*c_int),
                ("maxGridSize", 3*c_int),
                ("totalConstMem", c_size_t),
                ("major", c_int),
                ("minor", c_int),
                ("clockRate", c_int),
                ("textureAlignment", c_size_t)]

    def __str__(self):
        return """NVidia GPU Specifications:
    Name: %s
    Total global mem: %i
    Shared mem per block: %i
    Registers per block: %i
    Warp size: %i
    Mem pitch: %i
    Max threads per block: %i
    Max treads dim: (%i, %i, %i)
    Max grid size: (%i, %i, %i)
    Total const mem: %i
    Compute capability: %i.%i
    Clock Rate (GHz): %f
    Texture alignment: %i
""" % (self.name, self.totalGlobalMem, self.sharedMemPerBlock,
       self.regsPerBlock, self.warpSize, self.memPitch,
       self.maxThreadsPerBlock, 
       self.maxThreadsDim[0], self.maxThreadsDim[1], self.maxThreadsDim[2],
       self.maxGridSize[0], self.maxGridSize[1], self.maxGridSize[2],
       self.totalConstMem, self.major, self.minor,
       float(self.clockRate)/1.0e6, self.textureAlignment)


# cudaError_t
error_t = c_int

class UncastablePointerError(Exception):
    pass

#----------------------------------------------------------------------------
# Utility functions
#----------------------------------------------------------------------------

def _checkDeviceNumber(device):
    assert isinstance(device, int), "device number must be an int"
    assert device >= 0, "device number must be greater than 0"
    assert device < 2**8-1, "device number must be < 255" 

def _checkSizet(name, value):
    assert isinstance(value, (int, long)), "%s must be an int or long" % name

def _castToVoidp(name, value):
    if isinstance(value, c_void_p):
        return value
    else:
        try:
            castValue = cast(value, c_void_p)
        except:
            raise UncastablePointerError("%s can't be cast to void *" % name)
        else:
            return castValue

def _checkInt(name, value):
    assert isinstance(value, int), "%s must be an int" % name

#----------------------------------------------------------------------------
# D.1 Device Management
#----------------------------------------------------------------------------


# cudaGetDeviceCount
_cudaGetDeviceCount = libcudart.cudaGetDeviceCount
_cudaGetDeviceCount.restype = error_t
_cudaGetDeviceCount.argtypes = [POINTER(c_int)]

def getDeviceCount():
    i = c_int()
    status = _cudaGetDeviceCount(byref(i))
    _checkCudaStatus(status)
    return i.value


# cudaGetDeviceProperties
_cudaGetDeviceProperties = libcudart.cudaGetDeviceProperties
_cudaGetDeviceProperties.restype = error_t
_cudaGetDeviceProperties.argtypes = [POINTER(DeviceProp), c_int]

def getDeviceProperties(device):
    _checkDeviceNumber(device)
    props = DeviceProp()
    status = _cudaGetDeviceProperties(byref(props), device)
    _checkCudaStatus(status)
    return props


# cudaChooseDevice
_cudaChooseDevice = libcudart.cudaChooseDevice
_cudaChooseDevice.restype = error_t
_cudaChooseDevice.argtypes = [POINTER(c_int), POINTER(DeviceProp)]

def chooseDevice(deviceProp):
    assert isinstance(deviceProp, DeviceProp), "deviceProp be be a DeviceProp instance"
    i = c_int()
    status = _cudaChooseDevice(byref(i), byref(deviceProp))
    _checkCudaStatus(status)
    return i.value    


# cudaSetDevice
_cudaSetDevice = libcudart.cudaSetDevice
_cudaSetDevice.restype = error_t
_cudaSetDevice.argtypes = [c_int]

def setDevice(device):
    _checkDeviceNumber(device)
    status = _cudaSetDevice(device)
    _checkCudaStatus(status)


# cudaGetDevice
_cudaGetDevice = libcudart.cudaGetDevice
_cudaGetDevice.restype = error_t
_cudaGetDevice.argtypes = [POINTER(c_int)]

def getDevice():
    i = c_int()
    status = _cudaGetDevice(byref(i))
    _checkCudaStatus(status)
    return i.value


#----------------------------------------------------------------------------
# D.2 Thread Management
#----------------------------------------------------------------------------

# cudaThreadSynchronize
_cudaThreadSynchronize = libcudart.cudaThreadSynchronize
_cudaThreadSynchronize.restype = error_t
_cudaThreadSynchronize.argtypes = []

def threadSynchronize():
    status = _cudaThreadSynchronize()
    _checkCudaStatus(status)


# cudaThreadExit
_cudaThreadExit = libcudart.cudaThreadExit
_cudaThreadExit.restype = error_t
_cudaThreadExit.argtypes = []

def threadExit():
    status = _cudaThreadExit()
    _cudaThreadExit(status)


#----------------------------------------------------------------------------
# D.3 Memory Management
#----------------------------------------------------------------------------

# cudaMalloc
_cudaMalloc = libcudart.cudaMalloc
_cudaMalloc.restype = error_t
_cudaMalloc.argtypes = [POINTER(c_void_p), c_size_t]

def malloc(count):
    _checkSizet('count', count)
    assert count > 0, "count must be > 0"
    devPtr = c_void_p()
    status = _cudaMalloc(byref(devPtr), count)
    _checkCudaStatus(status)
    return devPtr


# cudaMallocPitch
_cudaMallocPitch = libcudart.cudaMallocPitch
_cudaMallocPitch.restype = error_t
_cudaMallocPitch.argtypes = [POINTER(c_void_p), POINTER(c_size_t),
    c_size_t, c_size_t]

def mallocPitch(widthInBytes, height):
    _checkSizet('widthInBytes', widthInBytes)
    _checkSizet('height', height)
    assert widthInBytes > 0 and height > 0, "widthInBytes and height must be > 0"
    devPtr = c_void_p()
    pitch = c_size_t()
    status = _cudaMallocPitch(byref(devPtr), byref(pitch), widthInBytes, height)
    _checkCudaStatus(status)
    return devPtr, pitch.value


# cudaFree
_cudaFree = libcudart.cudaFree
_cudaFree.restype = error_t
_cudaFree.argtypes = [c_void_p]

def free(devPtr):
    devPtr = _castToVoidp('devPtr', devPtr)
    status = _cudaFree(devPtr)
    _checkCudaStatus(status)

# CudaArray
class CudaArray(Structure):
    _fields = []

# enum for the f attribute of ChannelFormatDesc
cudaChannelFormatKindSigned = 0
cudaChannelFormatKindUnsigned = 1
cudaChannelFormatKindFloat = 2
 
# cudaChannelFormatDesc
class ChannelFormatDesc(Structure):
    _fields_ = [("x", c_int),
                ("y", c_int),
                ("z", c_int),
                ("w", c_int),
                ("f", c_int)]

# cudaMallocArray
_cudaMallocArray = libcudart.cudaMallocArray
_cudaMallocArray.restype = error_t
_cudaMallocArray.argtypes = [POINTER(POINTER(CudaArray)), POINTER(ChannelFormatDesc), 
    c_size_t, c_size_t]

def mallocArray(channelFormatDesc, width, height):
    _checkSizet('width', width)
    _checkSizet('height', height)
    assert isinstance(channelFormatDesc, ChannelFormatDesc), "Invalid type for channelFormatDesc"
    cudaArrayPointer = POINTER(CudaArray)()
    status = _cudaMallocArray(byref(cudaArrayPointer), byref(channelFormatDesc), width, height)
    _checkCudaStatus(status)
    return cudaArrayPointer

# cudaFreeArray
_cudaFreeArray = libcudart.cudaFreeArray
_cudaFreeArray.restype = error_t
_cudaFreeArray.argtypes = [POINTER(CudaArray)]

def freeArray(arrayPtr):
    assert isinstance(arrayPtr, POINTER(CudaArray)), "arrayPtr must be a POINTER(CudaArray)"
    status = _cudaFreeArray(arrayPtr)
    _checkCudaStatus(status)    

# cudaMallocHost
_cudaMallocHost = libcudart.cudaMallocHost
_cudaMallocHost.restype = error_t
_cudaMallocHost.argtypes = [POINTER(c_void_p), c_size_t]

def mallocHost(count):
    _checkSizet('count', count)
    assert count > 0, "count must be > 0"
    hostPtr = c_void_p()
    status = _cudaMallocHost(byref(hostPtr), count)
    _checkCudaStatus(status)
    return hostPtr    


# cudaFreeHost
_cudaFreeHost = libcudart.cudaFreeHost
_cudaFreeHost.restype = error_t
_cudaFreeHost.argtypes = [c_void_p]

def freeHost(hostPtr):
    hostPtr = _castToVoidp('hostPtr', hostPtr)
    status = _cudaFreeHost(hostPtr)
    _checkCudaStatus(status)


# cudaMemset
_cudaMemset = libcudart.cudaMemset
_cudaMemset.restype = error_t
_cudaMemset.argtypes = [c_void_p, c_int, c_size_t]

def memset(devPtr, value, count):
    devPtr = _castToVoidp('devPtr', devPtr)
    _checkInt('value', value)
    _checkSizet('count', count)
    status = _cudaMemset(devPtr, value, count)
    _checkCudaStatus(status)


# cudaMemset2D
_cudaMemset2D = libcudart.cudaMemset2D
_cudaMemset2D.restype = error_t
_cudaMemset2D.argtypes = [c_void_p, c_size_t, c_int, c_size_t, c_size_t]

def memset2D(devPtr, pitch, value, width, height):
    devPtr = _castToVoidp('devPtr', devPtr)
    _checkSizet('pitch', pitch)
    _checkInt('value', value)
    _checkSizet('width', width)
    _checkSizet('height', height)
    status = _cudaMemset2D(devPtr, pitch, value, width, height)
    _checkCudaStatus(status)
        

# cudaMemcpy
_cudaMemcpy = libcudart.cudaMemcpy
_cudaMemcpy.restype = error_t
_cudaMemcpy.argtypes = [c_void_p, c_void_p, c_size_t, c_int]

def cudaMemcpy(dstPtr, srcPtr, count, kind):
    dstPtr = _castToVoidp('dstPtr', dstPtr)
    srcPtr = _castToVoidp('srcPtr', srcPtr)
    _checkVoidp('srcPtr', srcPtr)
    _checkSizet('count', count)
    _checkInt('kind', kind)
    assert kind in range(4), "kind must be in the set (0,1,2,3)"
    status = _cudaMemcpy(dstPtr, srcPtr, count, kind)
    _checkCudaStatus(status)


# cudaMemcpy2D
_cudaMemcpy2D = libcudart.cudaMemcpy2D
_cudaMemcpy2D.restype = error_t
_cudaMemcpy2D.argtypes = [c_void_p, c_size_t, c_void_p, c_size_t, 
    c_size_t, c_size_t, c_int]

def cudaMemcpy(dst, dpitch, src, spitch, width, height, kind):
    dst = _castToVoidp('dst', dst)
    _checkSizet('dpitch', dpitch)
    src = _castToVoidp('src', src)
    _checkSizet('spitch', spitch)
    _checkSizet('width', width)
    _checkSizet('height', height)
    _checkInt('kind', kind)
    assert kind in range(4), "kind must be in the set (0,1,2,3)"
    status = _cudaMemcpy2D(dst, src, count, kind)
    _checkCudaStatus(status)

# cudaMemcpyToArray
# cudaMmcpy2DToArray
# cudaMemcpyFromArray
# cudaMemcpy2DFromArray
# cudaMemcpyArrayToArray
# cudaMemcpy2DArrayToArray

# These are templated...
# cudaMemcpyToSymbol
# cudaMemcpyFromSymbol
# cudaGetSymbolAddress
# cudaGetSymbolSize

#----------------------------------------------------------------------------
# D.4 Texture Reference Management
#----------------------------------------------------------------------------

# cudaCreateChannelDesc
_cudaCreateChannelDesc = libcudart.cudaCreateChannelDesc
_cudaCreateChannelDesc.restype = ChannelFormatDesc
_cudaCreateChannelDesc.argtypes = [c_int, c_int, c_int, c_int, c_int]

def createChannelDesc(x, y, z, w, f):
    _checkInt('x',x)
    _checkInt('y',y)
    _checkInt('z',z)
    _checkInt('f',f)
    assert f in [0,1,2], "The format f must be 0,1 or 2."
    cd = _cudaCreateChannelDesc(x,y,z,w,f)
    return cd


# cudaGetChannelDesc
_cudaGetChannelDesc = libcudart.cudaGetChannelDesc
_cudaGetChannelDesc.restype = cudaGetChannelDesc
_cudaGetChannelDesc.argtypes = [c_int, c_int, c_int, c_int, c_int]

def getChannelDesc(array):
    assert isinstance(array, CudaArray), "array must be a CudaArray struct."
    cd = ChannelFormatDesc()
    status = _cudaGetChannelDesc(byref(cd), byref(array))
    _checkCudaStatus(status)
    return cd

# These appear to require templated code...
# cudaGetTextureReference
# cudaBindTexture
# cudaBindTextureToArray
# cudaUnbindTexture
# cudaGetTextureAlignmentOffset

#----------------------------------------------------------------------------
# D.5 Execution Control
#----------------------------------------------------------------------------

# cudaConfigureCall
# cudaLaunch
# cudaSetupArgument

#----------------------------------------------------------------------------
# D.6 OpenGL Interoperability
#----------------------------------------------------------------------------

# cudaGLRegisterBufferObject
# cudaGLMapBufferObject
# cudaGLUnmapBufferObject
# cudaGLUnregisterBufferObject

#----------------------------------------------------------------------------
# D.7 Direct3D Interoperability
#----------------------------------------------------------------------------

# cudaD3D9Begin
# cudaD3D9End
# cudaD3D9RegisterVertexBuffer
# cudaD3D9MapVertexBuffer
# cudaD3D9UnmapVertexBuffer
# cudaD3D9UnregisterVertexBuffer

#----------------------------------------------------------------------------
# D.8 Error Handling
#----------------------------------------------------------------------------

# cudaGetLastError
_cudaGetLastError = libcudart.cudaGetLastError
_cudaGetLastError.restype = error_t
_cudaGetLastError.argtypes = []

def cudaGetLastError():
    return _cudaGetLastError()

# cudaGetErrorString
_cudaGetErrorString = libcudart.cudaGetErrorString
_cudaGetErrorString.restype = c_char_p
_cudaGetErrorString.argtypes = [error_t]

def getErrorString(value):
    return _cudaGetErrorString(value)
