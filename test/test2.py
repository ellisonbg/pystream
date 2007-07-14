import cublas
from ctypes import *
import numpy
cublas.cublasInit()
dp = cublas.cublasAlloc(10,c_int)
a = numpy.arange(10)
cublas.cublasSetVector(10,c_int,a.ctypes.data,1,dp,1)
a.fill(0)
cublas.cublasGetVector(10,c_int,dp,1,a.ctypes.data,1)
print a
