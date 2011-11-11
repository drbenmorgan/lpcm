'''
Created on 28 Oct 2011

@author: droythorne
'''
from droythorne.lpcm.lpc import LPCImpl
from droythorne.lpcm.lpcDiagnostics import LPCResiduals
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.mlab import griddata
from mpl_toolkits.mplot3d import axes3d, Axes3D
from numpy.core.function_base import linspace
from numpy.core.numeric import array, arange
from numpy.ma.core import sin, cos, zeros
from random import gauss
from scipy.constants.constants import pi
from scipy.interpolate.fitpack2 import UnivariateSpline
from scipy.interpolate.rbf import Rbf
import matplotlib.pyplot as plt

def plot1():
  fig1 = plt.figure()
  x = map(lambda x: x + gauss(0,0.005), arange(-1,1,0.005))
  y = map(lambda x: x + gauss(0,0.005), arange(-1,1,0.005))
  z = map(lambda x: x + gauss(0,0.005), arange(-1,1,0.005))
  line = array(zip(x,y,z))
  lpc = LPCImpl(h = 0.2, convergence_at = 0.001, mult = 2)
  lpc_curve = lpc.lpc(line)
  ax = Axes3D(fig1)
  ax.set_title('testNoisyLine1')
  curve = lpc_curve[0]['save_xd']
  ax.scatter(curve[:,0],curve[:,1],curve[:,2],c = 'red')
  return fig1
def plot2():
  fig2 = plt.figure()
  x = map(lambda x: x + gauss(0,0.002), arange(-1,1,0.001))
  y = map(lambda x: x + gauss(0,0.002), arange(-1,1,0.001))
  z = map(lambda x: x + gauss(0,0.02), arange(-1,1,0.001))
  line = array(zip(x,y,z))
  lpc = LPCImpl(h = 0.2, mult = 2)
  lpc_curve = lpc.lpc(line)
  ax = Axes3D(fig2)
  ax.set_title('testNoisyLine2')
  curve = lpc_curve[0]['save_xd']
  ax.scatter(curve[:,0],curve[:,1],curve[:,2])
def helixNonRandom():
  #Parameterise a helix (no noise)
  fig3 = plt.figure()
  t = arange(-1,1,0.001)
  x = (1 - t*t)*sin(4*pi*t)
  y = (1 - t*t)*cos(4*pi*t)
  z = t
  line = array(zip(x,y,z))
  lpc = LPCImpl(h = 0.15, t0 = 0.2, mult = 2, it = 500, scaled = False)
  lpc_curve = lpc.lpc(line)
  ax = Axes3D(fig3)
  ax.set_title('helixNonRandom')
  curve = lpc_curve[0]['save_xd']
  ax.scatter(x,y,z, c = 'red')
  ax.plot(curve[:,0],curve[:,1],curve[:,2])

def helixRandom():
  #Parameterise a helix (no noise)
  fig4 = plt.figure()
  t = arange(-1,1,0.001)
  x = map(lambda x: x + gauss(0,0.01), (1 - t*t)*sin(4*pi*t))
  y = map(lambda x: x + gauss(0,0.01), (1 - t*t)*cos(4*pi*t))
  z = map(lambda x: x + gauss(0,0.01), t)
  line = array(zip(x,y,z))
  lpc = LPCImpl(h = 0.15, t0 = 0.2, mult = 2, it = 500, scaled = False)
  lpc_curve = lpc.lpc(line)
  ax = Axes3D(fig4)
  ax.set_title('helixRandom')
  curve = lpc_curve[0]['save_xd']
  ax.scatter(x,y,z, c = 'red')
  ax.plot(curve[:,0],curve[:,1],curve[:,2])

def helixHeteroscedasticCrossingDemo():
  #Parameterise a helix (no noise)
  fig5 = plt.figure()
  t = arange(-1,1,0.001)
  x = map(lambda x: x + gauss(0,0.01 + 0.05*sin(8*pi*x)), (1 - t*t)*sin(4*pi*t))
  y = map(lambda x: x + gauss(0,0.01 + 0.05*sin(8*pi*x)), (1 - t*t)*cos(4*pi*t))
  z = map(lambda x: x + gauss(0,0.01 + 0.05*sin(8*pi*x)), t)
  line = array(zip(x,y,z))
  lpc = LPCImpl(h = 0.15, t0 = 0.1, mult = 2, it = 500, scaled = False)
  lpc_curve = lpc.lpc(line)
  ax = Axes3D(fig5)
  ax.set_title('helixHeteroscedasticWithCrossing')
  curve = lpc_curve[0]['save_xd']
  ax.scatter(x,y,z, c = 'red')
  ax.plot(curve[:,0],curve[:,1],curve[:,2])
  saveToPdf(fig5, '/tmp/helixHeteroscedasticWithCrossing.pdf')
  lpc.set_in_dict('cross', False, '_lpcParameters')
  fig6 = plt.figure()
  lpc_curve = lpc.lpc(line)
  ax = Axes3D(fig6)
  ax.set_title('helixHeteroscedasticWithoutCrossing')
  curve = lpc_curve[0]['save_xd']
  ax.scatter(x,y,z, c = 'red')
  ax.plot(curve[:,0],curve[:,1],curve[:,2])
  saveToPdf(fig6, '/tmp/helixHeteroscedasticWithoutCrossing.pdf')
  
def helixHeteroscedasticDiags():
  #Parameterise a helix (no noise)
  fig5 = plt.figure()
  t = arange(-1,1,0.0005)
  x = map(lambda x: x + gauss(0,0.001 + 0.001*sin(2*pi*x)**2), (1 - t*t)*sin(4*pi*t))
  y = map(lambda x: x + gauss(0,0.001 + 0.001*sin(2*pi*x)**2), (1 - t*t)*cos(4*pi*t))
  z = map(lambda x: x + gauss(0,0.001 + 0.001*sin(2*pi*x)**2), t)
  line = array(zip(x,y,z))
  lpc = LPCImpl(h = 0.1, t0 = 0.1, mult = 1, it = 500, scaled = False, cross = False)
  lpc_curve = lpc.lpc(line)
  ax = Axes3D(fig5)
  ax.set_title('helixHeteroscedastic')
  curve = lpc_curve[0]['save_xd']
  ax.scatter(x,y,z, c = 'red')
  ax.plot(curve[:,0],curve[:,1],curve[:,2])
  saveToPdf(fig5, '/tmp/helixHeteroscedastic.pdf')
  residuals_calc = LPCResiduals(line, tube_radius = 0.2, k = 20)
  residual_diags = residuals_calc.getPathResidualDiags(lpc_curve[0])
  fig6 = plt.figure()
  #plt.plot(lpc_curve[0]['lamb'][1:], residual_diags['line_seg_num_NN'], drawstyle = 'step', linestyle = '--')
  plt.plot(lpc_curve[0]['lamb'][1:], residual_diags['line_seg_mean_NN'])
  plt.plot(lpc_curve[0]['lamb'][1:], residual_diags['line_seg_std_NN'])
  saveToPdf(fig6, '/tmp/helixHeteroscedasticPathResiduals.pdf')
  #coverage_graph = residuals_calc.getCoverageGraph(lpc_curve[0], arange(0.01, .252, 0.05))
  #fig7 = plt.figure()
  #plt.plot(coverage_graph[0],coverage_graph[1])
  #saveToPdf(fig7, '/tmp/helixHeteroscedasticCoverage.pdf')
  residual_graph = residuals_calc.getGlobalResiduals(lpc_curve[0])
  fig8 = plt.figure()
  plt.plot(residual_graph[0], residual_graph[1])
  saveToPdf(fig8, '/tmp/helixHeteroscedasticResiduals.pdf')
  fig9 = plt.figure()
  plt.plot(range(len(lpc_curve[0]['lamb'])), lpc_curve[0]['lamb'])
def saveToPdf(fig, filename):
  pp = PdfPages(filename)
  pp.savefig(fig)
  pp.close() 

if __name__ == '__main__':
    
    #fig1 = plot1()
    #fig2 = plot2()
    #fig3 = helixNonRandom()
    #fig4 = helixRandom()
    fig5 = helixHeteroscedasticDiags()
    
    plt.show()
