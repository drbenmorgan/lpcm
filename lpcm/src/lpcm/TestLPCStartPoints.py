'''
Created on 17 Nov 2011

@author: droythorne
'''
from lpcm.lpcStartPoints import lpcRandomStartPoints, lpcMeanShift
from numpy.core.numeric import arange, array
from numpy.core.shape_base import hstack
from random import gauss
import unittest


class TestLPCStartPoints(unittest.TestCase):

    def setUp(self):
      pass
    def testlpcRandomStartPoints1(self):
      x = arange(0,1,0.1)
      y = arange(0,1,0.1)
      z = arange(0,1,0.1)
      line = array(zip(x,y,z))
      lpcRSP = lpcRandomStartPoints(line)
      print 'Generate 1 point, no explicitly defined points'
      sp = lpcRSP(1)
      print sp
    def testlpcMeanShift1(self):
      t = hstack((arange(-1,-0.1,0.005), arange(0.1, 1, 0.005)))
      x = map(lambda x: x + gauss(0,0.005), t)
      y = map(lambda x: x + gauss(0,0.005), t)
      z = map(lambda x: x + gauss(0,0.005), t)
      line = array(zip(x,y,z))
      lpcMS = lpcMeanShift(line)
      sp = lpcMS()
      print sp
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()