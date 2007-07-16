""" Testing a more Pythonic interface to the FFT libraries.

Still not working.
"""


from ctypes import POINTER, c_float

import numpy as np

from pystream.cudaarray import RawCudaArray, CudaArrayFromArray
from pystream import cufft


c_complex = c_float*2

class PlanCache(object):
    """ Simple object that maintains a certain number of plans.

    Old plans get destroyed.
    """

    def __init__(self, size=64):
        # XXX: is that a good size?
        self.size = size

        # The map of plan arguments to plan handles and back again.
        self.handle_map = {}
        self.handle_map_inverse = {}

        # The list of handles in order of most recent use.
        self.handles = []

    def lookup(self, dims, type_, batch=None):
        """ Look up a plan in the cache or create one.

        If we need to create a new plan and the cache is full, dump the one
        least-recently used.
        """
        plan = self.handle_map.get((dims, type_, batch), None)
        if plan is None:
            plan = self.createPlan(dims, type_, batch)
        else:
            # Freshen the list of handles to ensure we're on top.
            i = self.handles.index(plan)
            self.handles.insert(0, self.handles.pop(i))

        return plan

    def createPlan(self, dims, type_, batch=None):
        """ Create a new plan.
        """
        ndims = len(dims)
        if ndims not in (1, 2, 3):
            raise ValueError("only 1, 2, and 3 dimensions are supported")
        if batch is not None and ndims != 1:
            raise ValueError("batching is only supported with 1D FFTs")
        if ndims == 1:
            args = (dims[0], type_, batch)
        else:
            args = dims + (type_,)
        func = {
            1: cufft.cufftPlan1d,
            2: cufft.cufftPlan2d,
            3: cufft.cufftPlan3d,
        }[ndims]
        plan = func(*args)
        self.handle_map[args] = plan
        self.handle_map_inverse[id(plan)] = args
        self.handles.insert(0, plan)

        if len(self.handles) > self.size:
            # Pop a handle.
            old = self.handles.pop()
            oldargs = self.handle_map_inverse.pop(id(old))
            self.handle_map.pop(oldargs)
            cufft.cufftDestroy(old)

        return plan

    # XXX: __del__ ?

_plan_cache = PlanCache()


def fft(a, out=None):
    """ Do a 1D forward FFT.
    """
    a = np.ascontiguousarray(a)
    if np.issubdtype(a.dtype, np.complexfloating):
        a = a.astype(np.csingle)
        type_ = cufft.CUFFT_C2C
        ct = c_complex
    else:
        a = a.astype(np.single)
        type_ = cufft.CUFFT_R2C
        ct = c_float
    nx = a.shape[-1]
    if a.ndim > 1:
        # Batch up many 1D transforms.
        batch = np.size // nx
    else:
        batch = None
    plan = _plan_cache.lookup(a.shape, type_, batch)

    if out is None:
        out = np.empty(a.shape, np.csingle)
    else:
        if not isinstance(out.dtype , np.csingle):
            raise ValueError("output must be single-precision complex")
        if out.shape != a.shape:
            raise ValueError("output must have the same shape as the input")

    input = CudaArrayFromArray(a)
    output = RawCudaArray(a.size, np.csingle)
    args = (plan, input.data, output.data)
    if type_ == cufft.CUFFT_R2C:
        func = cufft.cufftExecR2C
    else:
        func = cufft.cufftExecC2C
        args += (cufft.CUFFT_FORWARD,)

    func(*args)

    # XXX: there's an extra temporary here.
    out.flat[:] = output.toArray()

    return out


if __name__ == '__main__':
    a = np.arange(16, dtype=np.csingle)
    f = fft(a)
    print f
    print np.fft.fft(a)