'''
Created on 17 Nov 2011

@author: droythorne
'''
from lpcm.lpcStartPoints import lpcRandomStartPoints, lpcMeanShift
from numpy.core.numeric import arange, array
from numpy.core.shape_base import hstack, vstack
from random import gauss
import unittest


class TestLPCStartPoints(unittest.TestCase):

    def setUp(self):
      x = arange(0,1,0.1)
      y = arange(0,1,0.1)
      z = arange(0,1,0.1)
      self._line = array(zip(x,y,z))
      
      t = hstack((arange(-1,-0.1,0.005), arange(1, 2, 0.005)))
      x = map(lambda x: x + gauss(0,0.005), t)
      y = map(lambda x: x + gauss(0,0.005), t)
      z = map(lambda x: x + gauss(0,0.005), t)
      self._line_cluster = array(zip(x,y,z))
      
      t = arange(-1,1,0.001)
      x = map(lambda x: x + gauss(0,0.02)*(1-x*x), t)
      y = map(lambda x: x + gauss(0,0.02)*(1-x*x), t)
      z = map(lambda x: x + gauss(0,0.02)*(1-x*x), t)
      line = array(zip(x,y,z))
      self._line_cluster_2_ = vstack((line, line + 3))
          
    def testlpcRandomStartPoints1(self):
      print 'Generate 1 random start point from line, with no explicitly defined start points'
      lpcRSP = lpcRandomStartPoints()
      sp = lpcRSP(self._line, 1)
      print sp
      
    def testlpcMeanShift1(self):
      print 'Generate start points based on ms_h: 0.5, ms_sub: 30 and no explicitly defined cluster seed points'
      lpcMS = lpcMeanShift(ms_h = 0.5)
      sp = lpcMS(self._line_cluster)
      print sp
      
    def testlpcMeanShift2Fail(self):
      print 'Generate start points based on ms_h: 0.5, ms_sub: 30 and 1 explicitly defined cluster seed point '
      print 'Raises error, no nearest neighbour within bandwidth - empty array raises error in sklearn BallTree' 
      lpcMS = lpcMeanShift( ms_h = 0.5)
      x0 = array([[0.5, 0.5, 0.5]])
      self.assertRaises(ValueError, lpcMS, self._line_cluster, None, x0)
    
    def testlpcMeanShift2(self):
      print 'Generate start points based on ms_h: 0.5, ms_sub: 30 and 1 explicitly defined cluster seed point '
      lpcMS = lpcMeanShift( ms_h = 0.5)
      x0 = array([[1.5, 1.5, 1.5]])
      sp = lpcMS(self._line_cluster, n = 10, x0 = x0)
      print sp
    def testlpcMeanShift3(self):
      print 'Generate start points based on ms_h: 0.5, ms_sub: 30, n = 10 and no explicitly defined cluster seed points'
      lpcMS = lpcMeanShift( ms_h = 0.5)
      sp = lpcMS(self._line_cluster, n = 10)
      print sp
    def testlpcMeanShift4(self):
      print 'Generate start points based on ms_h: 0.5, ms_sub: 30, 1 explicitly defined cluster seed point and n = 10'
      lpcMS = lpcMeanShift( ms_h = 0.5)
      sp = lpcMS(self._line_cluster, x0 = [[1.5, 1.5, 1.5]] , n = 10)
      print sp
    def testlpcMeanShift5(self):
      lpcMS = lpcMeanShift(ms_h = 1)
      sp = lpcMS(self._line_cluster_2_, n = None)
      print sp  
      
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()