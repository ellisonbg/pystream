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

# CudaArray
class CudaArray(Structure):
    _fields = []

# enum for the f attribute of ChannelFormatDesc
channelFormatKindSigned = 0
channelFormatKindUnsigned = 1
channelFormatKindFloat = 2
 
# cudaChannelFormatDesc
class ChannelFormatDesc(Structure):

    def __init__(self, x=32, y=0, z=0, w=0, f=channelFormatKindSigned):
        self.x = x
        self.y = y
        self.z = z
        self.w = w
        self.f = f

    _fields_ = [("x", c_int),
                ("y", c_int),
                ("z", c_int),
                ("w", c_int),
                ("f", c_int)]

# cudaError_t
error_t = c_int

class UncastablePointerError(Exception):
    pass

class InvalidDim3(Exception):
    pass

# struct dim3
# {
#     unsigned int x, y, z;
# #if defined(__cplusplus)
#     dim3(unsigned int x = 1, unsigned int y = 1, unsigned int z = 1) : x(x), y(y), z(z) {}
#     dim3(uint3 v) : x(v.x), y(v.y), z(v.z) {}
#     operator uint3(void) { uint3 t; t.x = x; t.y = y; t.z = z; return t; }
# #endif /* __cplusplus */
# };
class dim3(Structure):
    
    def __init__(self, x, y=1, z=1):
        self.x = x
        self.y = y
        self.z = z
    
    _fields_ = [('x', c_uint),
                ('y', c_uint),
                ('z', c_uint)]


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

def _checkPointerCudaArray(name, value):
    assert isinstance(value, POINTER(CudaArray)), "%s must be a POINTER(CudaArray)" % name

def _checkMemcpyKind(name, value):
    assert value in range(4), "%s must be in the set (0,1,2,3)" % name

def _checkCharp(name, value):
    assert isinstance(value, string), "%s must be a string" % name

def _convertToDim3(t):
    assert isinstance(t, (list, tuple, int)), "type must be a list/tuple/int"
    assert len(t) in (1,2,3), "a dim3 must have length of 1,2 or 3"
    if isinstance(t, int):
        d3 = dim3(t)
        return d3
    else:
        if len(t)==1:
            d3 = dim3(t[0])
        elif len(t)==2:
            d3 = dim3(t[0], t[1])
        elif len(t)==3:
            d3 = dim3(t[0], t[1], t[2])
        else:
            raise InvalidDim3("%r could not be converted to a dim3 object" % t)
        return d3
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

def malloc(count, ctype=None):
    _checkSizet('count', count)
    assert count > 0, "count must be > 0"
    devPtr = c_void_p()
    status = _cudaMalloc(byref(devPtr), count)
    _checkCudaStatus(status)
    if ctype is not None:
        # Cast it to the appropriate pointer type.
        devPtr = cast(devPtr, POINTER(ctype))
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
    _checkPointerCudaArray('arrayPtr', arrayPtr)
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
    # Don't check value as it could be a 32bit hex number, which is not always an int.
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

def memcpy(dstPtr, srcPtr, count, kind):
    dstPtr = _castToVoidp('dstPtr', dstPtr)
    srcPtr = _castToVoidp('srcPtr', srcPtr)
    _checkSizet('count', count)
    _checkInt('kind', kind)
    _checkMemcpyKind('kind', kind)
    status = _cudaMemcpy(dstPtr, srcPtr, count, kind)
    _checkCudaStatus(status)


# cudaMemcpy2D
_cudaMemcpy2D = libcudart.cudaMemcpy2D
_cudaMemcpy2D.restype = error_t
_cudaMemcpy2D.argtypes = [c_void_p, c_size_t, c_void_p, c_size_t, 
    c_size_t, c_size_t, c_int]

def memcpy2D(dst, dpitch, src, spitch, width, height, kind):
    dst = _castToVoidp('dst', dst)
    _checkSizet('dpitch', dpitch)
    src = _castToVoidp('src', src)
    _checkSizet('spitch', spitch)
    _checkSizet('width', width)
    _checkSizet('height', height)
    _checkInt('kind', kind)
    _checkMemcpyKind('kind', kind)
    status = _cudaMemcpy2D(dst, src, count, kind)
    _checkCudaStatus(status)

# cudaMemcpyToArray
# cudaError_t cudaMemcpyToArray(struct cudaArray* dstArray, 
#                               size_t dstX, size_t dstY, 
#                               const void* src, size_t count, 
#                               enum cudaMemcpyKind kind); 
_cudaMemcpyToArray = libcudart.cudaMemcpyToArray
_cudaMemcpyToArray.restype = error_t
_cudaMemcpyToArray.argtypes = [POINTER(CudaArray), c_size_t, c_size_t,
    c_void_p, c_size_t, c_int]

def memcpyToArray(dstArray, dstX, dstY, src, count, kind):
    _checkPointerCudaArray('dstArray', dstArray)
    _checkSizet('dstX', dstX)
    _checkSizet('dstY', dstY)
    src = _castToVoidp('src', src)
    _checkSizet('count', count)
    _checkInt('kind', kind)
    _checkMemcpyKind('kind', kind)
    status = _cudaMemcpyToArray(dstArray, dstX, dstY, src, count, kind)
    _checkCudaStatus(status)

# cudaMmcpy2DToArray
# cudaError_t cudaMemcpy2DToArray(struct cudaArray* dstArray, 
#                                 size_t dstX, size_t dstY, 
#                                 const void* src, size_t spitch, 
#                                 size_t width, size_t height, enum cudaMemcpyKind kind);
_cudaMemcpy2DToArray = libcudart.cudaMemcpy2DToArray
_cudaMemcpy2DToArray.restype = error_t
_cudaMemcpy2DToArray.argtypes = [POINTER(CudaArray), c_size_t, c_size_t,
    c_void_p, c_size_t, c_size_t, c_size_t, c_int]

def memcpy2DToArray(dstArray, dstX, dstY, src, spitch, width, height, kind):
    _checkPointerCudaArray('dstArray', dstArray)
    _checkSizet('dstX', dstX)
    _checkSizet('dstY', dstY)
    src = _castToVoidp('src', src)
    _checkSizet('spitch', spitch)
    _checkSizet('width', width)
    _checkSizet('height', height)
    _checkInt('kind', kind)
    _checkMemcpyKind('kind', kind)
    status = _cudaMemcpy2DToArray(dstArray, dstX, dstY, src, spitch, width, height, kind)
    _checkCudaStatus(status)


# cudaMemcpyFromArray
# cudaError_t cudaMemcpyFromArray(void* dst, 
#                                 const struct cudaArray* srcArray, 
#                                 size_t srcX, size_t srcY, 
#                                 size_t count, 
#                                 enum cudaMemcpyKind kind); 
_cudaMemcpyFromArray = libcudart.cudaMemcpyFromArray
_cudaMemcpyFromArray.restype = error_t
_cudaMemcpyFromArray.argtypes = [c_void_p, POINTER(CudaArray), c_size_t, c_size_t,
    c_size_t, c_int]

def memcpyFromArray(dst, srcArray, srcX, srcY, count, kind):
    dst = _castToVoidp('dst', dst)
    _checkPointerCudaArray('srcArray', srcArray)
    _checkSizet('srcX', srcX)
    _checkSizet('srcY', srcY)
    _checkSizet('count', count)
    _checkInt('kind', kind)
    _checkMemcpyKind('kind', kind)
    status = _cudaMemcpyFromArray(dst, srcArray, srcX, srcY, count, kind)
    _checkCudaStatus(status)


# cudaMemcpy2DFromArray
# cudaError_t cudaMemcpy2DFromArray(void* dst, size_t dpitch, 
#                                  const struct cudaArray* srcArray, 
#                                   size_t srcX, size_t srcY, 
#                                   size_t width, size_t height, 
#                                   enum cudaMemcpyKind kind); 
_cudaMemcpy2DFromArray = libcudart.cudaMemcpy2DFromArray
_cudaMemcpy2DFromArray.restype = error_t
_cudaMemcpy2DFromArray.argtypes = [c_void_p, c_size_t, POINTER(CudaArray), c_size_t, c_size_t,
    c_size_t, c_size_t, c_int]

def memcpy2DFromArray(dst, dpitch, srcArray, srcX, srcY, width, height, kind):
    dst = _castToVoidp('dst', dst)
    _checkSizet('dpitch', dpitch)
    _checkPointerCudaArray('srcArray', srcArray)
    _checkSizet('srcX', srcX)
    _checkSizet('srcY', srcY)
    _checkSizet('width', width)
    _checkSizet('height', height)
    _checkInt('kind', kind)
    _checkMemcpyKind('kind', kind)
    status = _cudaMemcpy2DFromArray(dst, dpitch, srcArray, srcX, srcY, width, height, kind)
    _checkCudaStatus(status)


# cudaMemcpyArrayToArray
# cudaError_t cudaMemcpyArrayToArray(struct cudaArray* dstArray, 
#                                    size_t dstX, size_t dstY, 
#                                  const struct cudaArray* srcArray, 
#                                    size_t srcX, size_t srcY, 
#                                    size_t count, 
#                                    enum cudaMemcpyKind kind); 
_cudaMemcpyArrayToArray = libcudart.cudaMemcpyArrayToArray
_cudaMemcpyArrayToArray.restype = error_t
_cudaMemcpyArrayToArray.argtypes = [POINTER(CudaArray), c_size_t, c_size_t,
    POINTER(CudaArray), c_size_t, c_size_t, c_size_t, c_int]

def memcpyArrayToArray(dstArray, dstX, dstY, srcArray, srcX, srcY, count, kind):
    _checkPointerCudaArray('dstArray', dstArray)
    _checkSizet('dstX', dstX)
    _checkSizet('dstY', dstY)
    _checkPointerCudaArray('srcArray', srcArray)
    _checkSizet('srcX', srcX)
    _checkSizet('srcY', srcY)
    _checkSizet('count', count)
    _checkInt('kind', kind)
    _checkMemcpyKind('kind', kind)
    status = _cudaMemcpyArrayToArray(dstArray, dstX, dstY, srcArray, srcX, srcY, count, kind)
    _checkCudaStatus(status)


# cudaMemcpy2DArrayToArray
# cudaError_t cudaMemcpy2DArrayToArray(struct cudaArray* dstArray, 
#                                      size_t dstX, size_t dstY, 
#                                  const struct cudaArray* srcArray, 
#                                      size_t srcX, size_t srcY, 
#                                      size_t width, size_t height, enum cudaMemcpyKind kind);
_cudaMemcpy2DArrayToArray = libcudart.cudaMemcpy2DArrayToArray
_cudaMemcpy2DArrayToArray.restype = error_t
_cudaMemcpy2DArrayToArray.argtypes = [POINTER(CudaArray), c_size_t, c_size_t,
    POINTER(CudaArray), c_size_t, c_size_t, c_size_t, c_size_t, c_int]

def memcpy2DArrayToArray(dstArray, dstX, dstY, srcArray, srcX, srcY, width, height, kind):
    _checkPointerCudaArray('dstArray', dstArray)
    _checkSizet('dstX', dstX)
    _checkSizet('dstY', dstY)
    _checkPointerCudaArray('srcArray', srcArray)
    _checkSizet('srcX', srcX)
    _checkSizet('srcY', srcY)
    _checkSizet('width', width)
    _checkSizet('height', height)
    _checkInt('kind', kind)
    _checkMemcpyKind('kind', kind)
    status = _cudaMemcpy2DArrayToArray(dstArray, dstX, dstY, srcArray, srcX, srcY, 
        width, height, kind)
    _checkCudaStatus(status)


# cudaMemcpyToSymbol
# extern __host__ cudaError_t CUDARTAPI cudaMemcpyToSymbol(const char *symbol, const void *src, 
#     size_t count, size_t offset __dv(0), enum cudaMemcpyKind kind __dv(cudaMemcpyHostToDevice));
_cudaMemcpyToSymbol = libcudart.cudaMemcpyToSymbol
_cudaMemcpyToSymbol.restype = error_t
_cudaMemcpyToSymbol.argtypes = [c_char_p, c_void_p, c_size_t, c_size_t, c_int]

def memcpyToSymbol(symbol, src, count, offset=0, kind=memcpyHostToDevice):
    _checkCharp('symbol', symbol)
    src = _castToVoidp('src', src)
    _checkSizet('count', count)
    _checkSizet('offset', offset)
    _checkInt('kind', kind)
    _checkMemcpyKind('kind', kind)
    status = _cudaMemcpyToSymbol(symbol, src, count, offset, kind)
    _checkCudaStatus(status)


# cudaMemcpyFromSymbol
# extern __host__ cudaError_t CUDARTAPI cudaMemcpyFromSymbol(void *dst, const char *symbol, size_t count, 
#     size_t offset __dv(0), enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToHost));
_cudaMemcpyFromSymbol = libcudart.cudaMemcpyFromSymbol
_cudaMemcpyFromSymbol.restype = error_t
_cudaMemcpyFromSymbol.argtypes = [c_void_p, c_char_p, c_size_t, c_size_t, c_int]

def memcpyFromSymbol(dst, symbol, count, offset=0, kind=memcpyDeviceToHost):
    dst = _castToVoidp('dst', dst)
    _checkCharp('symbol', symbol)
    _checkSizet('count', count)
    _checkSizet('offset', offset)
    _checkInt('kind', kind)
    _checkMemcpyKind('kind', kind)
    status = _cudaMemcpyFromSymbol(dst, symbol, count, offset, kind)
    _checkCudaStatus(status)


# cudaGetSymbolAddress
# extern __host__ cudaError_t CUDARTAPI cudaGetSymbolAddress(void **devPtr, const char *symbol);
_cudaGetSymbolAddress = libcudart.cudaGetSymbolAddress
_cudaGetSymbolAddress.restype = error_t
_cudaGetSymbolAddress.argtypes = [POINTER(c_void_p), c_char_p]

def getSymbolAddress(symbol):
    _checkCharp('symbol', symbol)
    devPtr = c_void_p(0)
    status = _cudaGetSymbolAddress(byref(devPtr), symbol)
    _checkCudaStatus(status)
    return devPtr


# cudaGetSymbolSize
# extern __host__ cudaError_t CUDARTAPI cudaGetSymbolSize(size_t *size, const char *symbol);
_cudaGetSymbolSize = libcudart.cudaGetSymbolSize
_cudaGetSymbolSize.restype = error_t
_cudaGetSymbolSize.argtypes = [POINTER(c_size_t), c_char_p]

def getSymbolSize(symbol):
    _checkCharp('symbol', symbol)
    size = c_size_t(0)
    status = _cudaGetSymbolSize(byref(size), symbol)
    _checkCudaStatus(status)
    return size.contents.value

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
_cudaGetChannelDesc.restype = error_t
_cudaGetChannelDesc.argtypes = [POINTER(ChannelFormatDesc), POINTER(CudaArray)]

def getChannelDesc(arrayPtr):
    _checkPointerCudaArray('arrayPtr', arrayPtr)
    cd = ChannelFormatDesc()
    status = _cudaGetChannelDesc(byref(cd), arrayPtr)
    _checkCudaStatus(status)
    return cd

# cudaGetTextureReference
# extern __host__ cudaError_t CUDARTAPI cudaGetTextureReference(
#     const struct textureReference **texref, const char *symbol);

# cudaBindTexture
# extern __host__ cudaError_t CUDARTAPI cudaBindTexture(size_t *offset, 
#     const struct textureReference *texref, const void *devPtr, 
#     const struct cudaChannelFormatDesc *desc, size_t size __dv(UINT_MAX));

# cudaBindTextureToArray
# extern __host__ cudaError_t CUDARTAPI cudaBindTextureToArray(
#     const struct textureReference *texref, const struct cudaArray *array, 
#     const struct cudaChannelFormatDesc *desc);

# cudaUnbindTexture
# extern __host__ cudaError_t CUDARTAPI cudaUnbindTexture(
#     const struct textureReference *texref);

# cudaGetTextureAlignmentOffset
# extern __host__ cudaError_t CUDARTAPI cudaGetTextureAlignmentOffset(
#     size_t *offset, const struct textureReference *texref);


#----------------------------------------------------------------------------
# D.5 Execution Control
#----------------------------------------------------------------------------

# While these have been wrapped, I am not sure how they are used.  The
# documentation from NVIDIA seems very limited on the subject.

# cudaConfigureCall
# extern __host__ cudaError_t CUDARTAPI cudaConfigureCall(dim3 gridDim, 
#     dim3 blockDim, size_t sharedMem __dv(0), int tokens __dv(0));
_cudaConfigureCall = libcudart.cudaConfigureCall
_cudaConfigureCall.restype = error_t
_cudaConfigureCall.argtypes = [dim3, dim3, c_size_t, c_int]

def cudaConfigureCall(gridDim, blockDim, sharedMem=0, tokens=0):
    gd3 = _convertToDim3(gridDim)
    bd3 = _convertToDim3(blockDim)
    _checkSizet('sharedMem', sharedMem)
    _checkInt('tokens', tokens)
    status = _cudaConfigureCall(gd3, bd3, sharedMem, tokens)
    _checkCudaStatus(status)


# cudaSetupArgument
# extern __host__ cudaError_t CUDARTAPI cudaSetupArgument(const void *arg, 
#     size_t size, size_t offset);
_cudaSetupArgument = libcudart.cudaSetupArgument
_cudaSetupArgument.restype = error_t
_cudaSetupArgument.argtypes = [c_void_p, c_size_t, c_size_t]

def cudaSetupArgument(arg, size, offset):
    arg = _castToVoidp('arg', arg)
    _checkSizet('size', size)
    _checkSizet('offset', offset)
    status = _cudaSetupArgument(arg, size, offset)
    _checkCudaStatus(status)


# cudaLaunch
# extern __host__ cudaError_t CUDARTAPI cudaLaunch(const char *symbol);
_cudaLaunch = libcudart.cudaLaunch
_cudaLaunch.restype = error_t
_cudaLaunch.argtypes = [c_char_p]

def launch(symbol):
    _checkCharp('symbol', symbol)
    status = _cudaLaunch(symbol)
    _checkCudaStatus(status)

#----------------------------------------------------------------------------
# D.6 OpenGL Interoperability
#----------------------------------------------------------------------------
# Anyone want to wrap these?

# cudaGLRegisterBufferObject
# cudaGLMapBufferObject
# cudaGLUnmapBufferObject
# cudaGLUnregisterBufferObject

#----------------------------------------------------------------------------
# D.7 Direct3D Interoperability
#----------------------------------------------------------------------------
# Anyone running Windows want to wrap these?

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

def getLastError():
    return _cudaGetLastError()

# cudaGetErrorString
_cudaGetErrorString = libcudart.cudaGetErrorString
_cudaGetErrorString.restype = c_char_p
_cudaGetErrorString.argtypes = [error_t]

def getErrorString(value):
    return _cudaGetErrorString(value)
