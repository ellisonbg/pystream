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
from pystream import cudart
import random

#----------------------------------------------------------------------------
# Testing utilities
#----------------------------------------------------------------------------

class DataCreator(object):

    def createIntArray(self, size):
        t = c_int*size
        a = t()
        return a

    def createUIntArray(self, size):
        t = c_uint*size
        a = t()
        return a

    def createRandomIntArray(self, size):
        a = self.createIntArray(size)
        for i in range(size):
            a[i] = random.randint(-2**31,2**31-1)
        return a

    def createRandomUIntArray(self, size):
        a = self.createIntArray(size)
        for i in range(size):
            a[i] = random.randint(0,2**32-1)
        return a

    def isIntArrayEqual(self, a, b):
        if not len(a)==len(b): return False
        length = len(a)
        same = True
        for i in range(length):
            if not a[i] == b[i]: same = False
        return same

#----------------------------------------------------------------------------
# The TestCases
#----------------------------------------------------------------------------

class TestDevice(unittest.TestCase):
    
    def testBasic(self):
        c = cudart.getDeviceCount()
        self.assert_(isinstance(c,int) and c>=1)
        d = cudart.getDevice()
        print "Default device: ", d
        self.assert_(isinstance(d,int))
        cudart.setDevice(d)
        dp = cudart.getDeviceProperties(d)
        self.assert_(isinstance(dp,cudart.DeviceProp))
        print dp
        d2 = cudart.chooseDevice(dp)
        self.assertEquals(d, d2)    
    
    def testDeviceProps(self):
        d = cudart.getDevice()
        dp = cudart.getDeviceProperties(d)
        self.assert_(isinstance(dp.name,str))
        self.assert_(isinstance(dp.totalGlobalMem, long))
        self.assert_(isinstance(dp.sharedMemPerBlock, long))
        self.assert_(isinstance(dp.regsPerBlock, int))
        self.assert_(isinstance(dp.warpSize, int))
        self.assert_(isinstance(dp.memPitch, long))
        self.assert_(isinstance(dp.maxThreadsPerBlock, int))
        self.assert_(isinstance(dp.totalConstMem, long))
        self.assert_(isinstance(dp.major, int))
        self.assert_(isinstance(dp.minor, int))
        self.assert_(isinstance(dp.clockRate, int))
        self.assert_(isinstance(dp.textureAlignment, long))

class TestThread(unittest.TestCase):

    def testBasic(self):
        cudart.threadSynchronize()
        cudart.threadExit()

class TestMalloc(unittest.TestCase):
    
    def testMalloc(self):
        sizes = [2**n for n in range(24)]
        for s in sizes:
            ptr = cudart.malloc(s)
            cudart.free(ptr)

    def testMallocPitch(self):
        sizes = [2**n for n in range(12)]
        for width in sizes:
            for height in sizes:
                (ptr, pitch) = cudart.mallocPitch(width, height)
                cudart.free(ptr)

    def testMallocHost(self):
        sizes = [2**n for n in range(12)]
        for s in sizes:
            ptr = cudart.mallocHost(s)
            cudart.freeHost(ptr)
      
    def testMallocArray(self):
        b = (0,32)
        for y, z, w in [(y, z, w) for y in b for z in b for w in b if (y>z and z>w and y>w)]:
            for f in (0,1,2):
                cd = cudart.ChannelFormatDesc(x=32,y=y,z=z,w=w,f=f)
                sizes = [2**n for n in range(12)]
                for width in sizes:
                    for height in sizes:
                        arrayPtr = cudart.mallocArray(cfd, width, height)
                        cd2 = getChannelDesc(a)
                        self.assertEquals(cd.x,cf2.x)
                        self.assertEquals(cd.y,cf2.y)
                        self.assertEquals(cd.z,cf2.z)
                        self.assertEquals(cd.w,cf2.w)
                        self.assertEquals(cd.f,cf2.f)
                        cudart.freeArray(ptr)


class TestMemcpy(unittest.TestCase, DataCreator):

    def testMemcpy(self):
        for s in [2**k for k in range(10)]:
            a = self.createRandomIntArray(s)
            d_a = cudart.malloc(s)
            cudart.memcpy(d_a, a, s*sizeof(c_int), cudart.memcpyHostToDevice)
            b = self.createIntArray(s)
            cudart.memcpy(b, d_a, s*sizeof(c_int), cudart.memcpyDeviceToHost)
            self.assert_(self.isIntArrayEqual(a,b))
        
        


class TestMemset(unittest.TestCase, DataCreator):

    def testMemset(self):
        for nbytes in [2**k for k in range(10)]:
            nunits = nbytes/4
            ptr = cudart.malloc(nbytes)
            cudart.memset(ptr, 0xFFFFFFFF, nbytes)
            a = self.createUIntArray(nunits)
            cudart.memcpy(a, ptr, nbytes, cudart.memcpyDeviceToHost)
            for i in range(len(a)):
                self.assertEquals(a[i],2**32-1)
            cudart.free(ptr)





class TestTexture(unittest.TestCase):
    pass


class TestExecutionControl(unittest.TestCase):
    pass
        

class TestErrorHandling(unittest.TestCase):


    def testErrorStrings(self):
        # print "********************************************"
        # print "Listing error codes and messages:"
        for k in cudart.errorDict.keys():
            s = cudart.getErrorString(k)
            self.assert_(isinstance(s, str))
            # print k, s
        # print "End of listing of error codes and messages"
        # print "********************************************"

    def testLastError(self):
        cudart.threadExit()
        e = cudart.getLastError()
        self.assertEquals(e, 0)


if __name__ == '__main__':
    try:
        unittest.main()
    except SystemExit:
        pass
