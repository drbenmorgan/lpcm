
'''
Created on 28 Oct 2011

@author: droythorne
'''
from lpcm.lpc import LPCImpl
from lpcm.lpcDiagnostics import LPCResiduals
from lpcm.lpcStartPoints import lpcMeanShift
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import axes3d, Axes3D
from numpy.core.function_base import linspace
from numpy.core.numeric import array, arange
from numpy.core.shape_base import vstack
from numpy.lib.function_base import unique
from numpy.ma.core import sin, cos, zeros
from random import gauss
from scipy.constants.constants import pi
from itertools import cycle
import matplotlib.pyplot as plt

def plot1():
  fig1 = plt.figure()
  x = map(lambda x: x + gauss(0,0.005), arange(-1,1,0.005))
  y = map(lambda x: x + gauss(0,0.005), arange(-1,1,0.005))
  z = map(lambda x: x + gauss(0,0.005), arange(-1,1,0.005))
  line = array(zip(x,y,z))
  lpc = LPCImpl(h = 0.05, mult = 2, scaled = False)
  lpc_curve = lpc.lpc(X=line)
  ax = Axes3D(fig1)
  ax.set_title('testNoisyLine1')
  curve = lpc_curve[0]['save_xd']
  ax.scatter(curve[:,0],curve[:,1],curve[:,2],c = 'red')
  return fig1
def plot2():
  fig5 = plt.figure()
  x = map(lambda x: x + gauss(0,0.02)*(1-x*x), arange(-1,1,0.001))
  y = map(lambda x: x + gauss(0,0.02)*(1-x*x), arange(-1,1,0.001))
  z = map(lambda x: x + gauss(0,0.02)*(1-x*x), arange(-1,1,0.001))
  line = array(zip(x,y,z))
  lpc = LPCImpl(h = 0.05, mult = 2, it = 200, cross = False, scaled = False, convergence_at = 0.001)
  lpc_curve = lpc.lpc(X=line)
  ax = Axes3D(fig5)
  ax.set_title('testNoisyLine2')
  curve = lpc_curve[0]['save_xd']
  ax.scatter(x,y,z, c = 'red')
  ax.plot(curve[:,0],curve[:,1],curve[:,2])
  saveToPdf(fig5, '/tmp/testNoisyLine2.pdf')
  residuals_calc = LPCResiduals(line, tube_radius = 0.05, k = 10)
  residual_diags = residuals_calc.getPathResidualDiags(lpc_curve[0])
  fig6 = plt.figure()
  #plt.plot(lpc_curve[0]['lamb'][1:], residual_diags['line_seg_num_NN'], drawstyle = 'step', linestyle = '--')
  plt.plot(lpc_curve[0]['lamb'][1:], residual_diags['line_seg_mean_NN'])
  plt.plot(lpc_curve[0]['lamb'][1:], residual_diags['line_seg_std_NN'])
  saveToPdf(fig6, '/tmp/testNoisyLine2PathResiduals.pdf')
  coverage_graph = residuals_calc.getCoverageGraph(lpc_curve[0], arange(0.001, .102, 0.005))
  fig7 = plt.figure()
  plt.plot(coverage_graph[0],coverage_graph[1])
  saveToPdf(fig7, '/tmp/testNoisyLine2Coverage.pdf')
  residual_graph = residuals_calc.getGlobalResiduals(lpc_curve[0])
  fig8 = plt.figure()
  plt.plot(residual_graph[0], residual_graph[1])
  saveToPdf(fig8, '/tmp/testNoisyLine2Residuals.pdf')
  fig9 = plt.figure()
  plt.plot(range(len(lpc_curve[0]['lamb'])), lpc_curve[0]['lamb'])
  saveToPdf(fig9, '/tmp/testNoisyLine2PathLength.pdf')
def helixNonRandom():
  #Parameterise a helix (no noise)
  fig3 = plt.figure()
  t = arange(-1,1,0.001)
  x = (1 - t*t)*sin(4*pi*t)
  y = (1 - t*t)*cos(4*pi*t)
  z = t
  line = array(zip(x,y,z))
  lpc = LPCImpl(h = 0.15, t0 = 0.2, mult = 2, it = 100, scaled = False)
  lpc_curve = lpc.lpc(X=line)
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
  lpc = LPCImpl(h = 0.15, t0 = 0.2, mult = 2, it = 100, scaled = False)
  lpc_curve = lpc.lpc(X=line)
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
  lpc_curve = lpc.lpc(X=line)
  ax = Axes3D(fig6)
  ax.set_title('helixHeteroscedasticWithoutCrossing')
  curve = lpc_curve[0]['save_xd']
  ax.scatter(x,y,z, c = 'red')
  ax.plot(curve[:,0],curve[:,1],curve[:,2])
  saveToPdf(fig6, '/tmp/helixHeteroscedasticWithoutCrossing.pdf')
def twoDisjointLinesWithMSClustering():
 
  t = arange(-1,1,0.002)
  x = map(lambda x: x + gauss(0,0.02)*(1-x*x), t)
  y = map(lambda x: x + gauss(0,0.02)*(1-x*x), t)
  z = map(lambda x: x + gauss(0,0.02)*(1-x*x), t)
  line1 = array(zip(x,y,z))
  line = vstack((line1, line1 + 3))
  lpc = LPCImpl(start_points_generator = lpcMeanShift(ms_h = 1), h = 0.05, mult = None, it = 200, cross = False, scaled = False, convergence_at = 0.001)
  lpc_curve = lpc.lpc(X=line)
  #Plot results
  fig = plt.figure()
  ax = Axes3D(fig)
  labels = lpc._startPointsGenerator._meanShift.labels_
  labels_unique = unique(labels)
  cluster_centers = lpc._startPointsGenerator._meanShift.cluster_centers_
  n_clusters = len(labels_unique)
  colors = cycle('bgrcmyk')
  for k, col in zip(range(n_clusters), colors):
    cluster_members = labels == k
    cluster_center = cluster_centers[k]
    ax.scatter(line[cluster_members, 0], line[cluster_members, 1], line[cluster_members, 2], c = col, alpha = 0.1)
    ax.scatter([cluster_center[0]], [cluster_center[1]], [cluster_center[2]], c = 'b', marker= '^')
    curve = lpc_curve[k]['save_xd']
    ax.plot(curve[:,0],curve[:,1],curve[:,2], c = col, linewidth = 3)
  plt.show()
  print 'Done and dusted'
def helixHeteroscedasticDiags():
  #Parameterise a helix (no noise)
  fig5 = plt.figure()
  t = arange(-1,1,0.0005)
  x = map(lambda x: x + gauss(0,0.001 + 0.001*sin(2*pi*x)**2), (1 - t*t)*sin(4*pi*t))
  y = map(lambda x: x + gauss(0,0.001 + 0.001*sin(2*pi*x)**2), (1 - t*t)*cos(4*pi*t))
  z = map(lambda x: x + gauss(0,0.001 + 0.001*sin(2*pi*x)**2), t)
  line = array(zip(x,y,z))
  lpc = LPCImpl(h = 0.1, t0 = 0.1, mult = 1, it = 500, scaled = False, cross = False)
  lpc_curve = lpc.lpc(X=line)
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
  coverage_graph = residuals_calc.getCoverageGraph(lpc_curve[0], arange(0.01, .052, 0.01))
  fig7 = plt.figure()
  plt.plot(coverage_graph[0],coverage_graph[1])
  saveToPdf(fig7, '/tmp/helixHeteroscedasticCoverage.pdf')
  residual_graph = residuals_calc.getGlobalResiduals(lpc_curve[0])
  fig8 = plt.figure()
  plt.plot(residual_graph[0], residual_graph[1])
  saveToPdf(fig8, '/tmp/helixHeteroscedasticResiduals.pdf')
  fig9 = plt.figure()
  plt.plot(range(len(lpc_curve[0]['lamb'])), lpc_curve[0]['lamb'])
  saveToPdf(fig9, '/tmp/helixHeteroscedasticPathLength.pdf')

def lamuMuonTest():
  pass

def saveToPdf(fig, filename):
  pp = PdfPages(filename)
  pp.savefig(fig)
  pp.close() 

if __name__ == '__main__':
    
    #fig1 = plot1()
    #fig2 = plot2()
    #fig3 = helixNonRandom()
    #fig4 = helixRandom()
    #fig5 = helixHeteroscedasticDiags()
    fig6 = twoDisjointLinesWithMSClustering()
    #plt.show()
