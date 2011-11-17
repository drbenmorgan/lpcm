'''
Created on 10 Oct 2011

@author: droythorne
'''
from lpcm.lpc import LPCImpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from numpy.core.numeric import array, arange
from random import gauss
import unittest


class TestLPCImpl(unittest.TestCase):

  def setUp(self):
    pass
  
  def testParamInit_h(self):
    params = ({'h': [0.005,-1]}, {'h': -0.005}, {'h': '0.005'}, {'h': ['0.1', '0.2']})
    map(lambda params: self.assertRaises(NameError, LPCImpl, **params), params)
    
  #Check that having points on a hyperplane doesn't screw up eigenvector calculation 
  def testLPCCalculation1(self):
    lpc = LPCImpl(h = [0.7, 0.7, 0.7], it = 10)
    lpc.lpc(array([[1.,2.,3.], [4.,5.,6.], [7.,8.,9.]]))
    #add assertion for h, t0 modification
  
  #Test forward direction which should converge before 10 iterations
  def testFollowXOne(self):
    lpc = LPCImpl(h = [0.6, 0.7, 0.6], it = 10, convergence_at = 0.005)
    lpc.lpc(array([[.1,.2,.3], [.11,.22,.33], [.07,.18,.29]]))
  
  #Test both directions which should both converge before 10 iterations
  def testFollowXTwo(self):
    lpc = LPCImpl(h = [0.6, 0.7, 0.6], it = 10, convergence_at = 0.005, way = 'two')
    lpc.lpc(array([[.1,.2,.3], [.11,.22,.33], [.07,.18,.29]]))
  
  def testNoisyLine1(self):
    x = map(lambda x: x + gauss(0,0.002), arange(-1,1,0.001))
    y = map(lambda x: x + gauss(0,0.002), arange(-1,1,0.001))
    z = map(lambda x: x + gauss(0,0.02), arange(-1,1,0.001))
    line = array(zip(x,y,z))
    lpc = LPCImpl(h = 0.2, mult = 2)
    lpc_curve = lpc.lpc(line)
  
  def testNoisyLine2(self):
    x = map(lambda x: x + gauss(0,0.005), arange(-1,1,0.005))
    y = map(lambda x: x + gauss(0,0.005), arange(-1,1,0.005))
    z = map(lambda x: x + gauss(0,0.005), arange(-1,1,0.005))
    line = array(zip(x,y,z))
    lpc = LPCImpl(h = 0.2, convergence_at = 0.001, mult = 2)
    lpc_curve = lpc.lpc(line) 

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()