"""A simple python script showing segfaults on exit.

This script uses the ctypes packages to dynamically load the CUDA
libraries.  We have been using this approach successfully since the
0.8 release of CUDA and it has worked extremely well.  This script
will run out of the box with Python 2.5 (ctypes comes with Python 2.5)
and will also run on Python 2.4 if ctypes is downloaded and installed
separately.

I don't think we had this problem with CUDA 0.8.

Run the script by doing:

$ python test_segfault.py
"""

from ctypes import *

# Uncomment a particular library to see its effect.  It is important
# to note that even though these segfault on exit, they can be 
# used without problems in the meantime.

# Either of these cause a segfault on exit.
#libcublas = cdll.LoadLibrary('libcublas.so')
#libcufft = cdll.LoadLibrary('libcufft.so')

# This does not segfault on exit.
libcudart = cdll.LoadLibrary('libcudart.so')
