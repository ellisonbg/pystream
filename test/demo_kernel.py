""" Nonworking test of the kernel launcher.

Read the comments below for building the kernel. It uses an example kernel from
the CUDA SDK.
"""

import atexit
import ctypes

import numpy as np

from pystream import cudaarray, cudart, kernels


BLOCK_SIZE = 16
WA = (3 * BLOCK_SIZE) # Matrix A width
HA = (5 * BLOCK_SIZE) # Matrix A height
WB = (8 * BLOCK_SIZE) # Matrix B width
HB = WA  # Matrix B height
WC = WB  # Matrix C width 
HC = HA  # Matrix C height


# Create this with the following commands from the CUDA SDK projects/matrixMul
# directory:
#   $ nvcc -Xcompiler="-fPIC" -c -o matrixMul_kernels.cu_o matrixMul_kernels.cu
#   $ g++ -shared -L/usr/local/cuda/lib -lcudart -lcuda -o libmatrixMul.so matrixMul_kernels.cu_o
#   $ cp libmatrixMul.so ~/src/pystream/test/
print 'Loading kernel'
dll = ctypes.cdll.LoadLibrary("./libmatrixMul.so")
# Register the finalizer. This seems to get rid of the segfault-on-exit that
# I've been seeing.
atexit.register(dll._fini)
matrixMul = kernels.KernelGetter(dll)


nA = np.random.random(size=(HA, WA)).astype(np.float32)
nB = np.random.random(size=(HB, WB)).astype(np.float32)

print 'Allocating arrays'
dA = cudaarray.CudaArrayFromArray(nA)
dB = cudaarray.CudaArrayFromArray(nB)
dC = cudaarray.RawCudaArray(HC*WC, dtype=np.float32)

threads = (BLOCK_SIZE, BLOCK_SIZE)
grid = (WC // BLOCK_SIZE, HC // BLOCK_SIZE)

print 'Calling kernel'
Mul = matrixMul.matrixMul(threads, grid)
Mul(dC.data, dA.data, dB.data, WA, WB)

print 'Collecting results'
print '  This currently causes an exception.'
nC = dC.toArray()
nC.shape = (HC, WC)

print 'Freeing data'
dA.free()
dB.free()
dC.free()

print 'Calculating error'
print
goldC = np.dot(nA, nB)
err = nC - goldC
print 'L2 err: %r' % np.linalg.norm(err, 2)
print 'L1 err: %r' % np.linalg.norm(err, 1)
print 'Linf err: %r' % np.linalg.norm(err, np.inf)
print 'Lfro err: %r' % np.linalg.norm(err, 'fro')
