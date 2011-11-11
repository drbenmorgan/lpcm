'''
Created on 28 Oct 2011

@author: droythorne
'''
from droythorne.lpcm.lpc import LPCImpl
from droythorne.lpcm.lpcDiagnostics import LPCResiduals
from numpy.core.numeric import array
from numpy.ma.core import arange
from random import gauss
import unittest


class TestLPCResiduals(unittest.TestCase):


  def testNoisyLine1Residuals(self):
    x = map(lambda x: x + gauss(0,0.005), arange(-1,1,0.005))
    y = map(lambda x: x + gauss(0,0.005), arange(-1,1,0.005))
    z = map(lambda x: x + gauss(0,0.005), arange(-1,1,0.005))
    line = array(zip(x,y,z))
    lpc = LPCImpl(h = 0.2, convergence_at = 0.0005, it = 500, mult = 2)
    lpc_curve = lpc.lpc(line) 
    residuals_calc = LPCResiduals(line, tube_radius = 0.1)
    residual_diags = residuals_calc.getPathResidualDiags(lpc_curve[0])
    
  def testNoisyLine2Residuals(self):
    #contains data that gets more scattered at each end of the line
    x = map(lambda x: x + gauss(0,0.005 + 0.3*x*x), arange(-1,1,0.005))
    y = map(lambda x: x + gauss(0,0.005 + 0.3*x*x), arange(-1,1,0.005))
    z = map(lambda x: x + gauss(0,0.005 + 0.3*x*x), arange(-1,1,0.005))
    line = array(zip(x,y,z))
    lpc = LPCImpl(h = 0.05, convergence_at = 0.001, it = 100, mult = 2)
    lpc_curve = lpc.lpc(line) 
    residuals_calc = LPCResiduals(line, tube_radius = 1)
    residual_diags = residuals_calc.getPathResidualDiags(lpc_curve[0])
    
  def testCoverage(self):
    x = map(lambda x: x + gauss(0,0.005 + 0.3*x*x), arange(-1,1,0.005))
    y = map(lambda x: x + gauss(0,0.005 + 0.3*x*x), arange(-1,1,0.005))
    z = map(lambda x: x + gauss(0,0.005 + 0.3*x*x), arange(-1,1,0.005))
    line = array(zip(x,y,z))
    lpc = LPCImpl(h = 0.05, convergence_at = 0.0001, it = 100, mult = 2)
    lpc_curve = lpc.lpc(line)
    residuals_calc = LPCResiduals(line, tube_radius = 1)
  #  coverage_graph = residuals_calc.getCoverageGraph(lpc_curve[0], arange(0.01, 1.01, 0.1))
    
  def testResiduals(self):
    x = map(lambda x: x + gauss(0,0.005 + 0.3*x*x), arange(-1,1,0.005))
    y = map(lambda x: x + gauss(0,0.005 + 0.3*x*x), arange(-1,1,0.005))
    z = map(lambda x: x + gauss(0,0.005 + 0.3*x*x), arange(-1,1,0.005))
    line = array(zip(x,y,z))
    lpc = LPCImpl(h = 0.05, convergence_at = 0.0001, it = 100, mult = 2)
    lpc_curve = lpc.lpc(line)
    residuals_calc = LPCResiduals(line, tube_radius = 1)
    residuals_graph = residuals_calc.getGlobalResiduals(lpc_curve[0])
    
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testLPCResiduals']
    unittest.main()