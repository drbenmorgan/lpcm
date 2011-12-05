'''
Created on 22 Nov 2011

@author: droythorne
'''
from itertools import cycle
from mpl_toolkits.mplot3d import axes3d, Axes3D
import matplotlib.pyplot as plt

class lpcDynamicBinnedResiduals():
  '''
  Generates a 1 dimensional histogram from a 2 dimensional, n * 2 array, where values in the the second column
  are a function of those  in the first (which are irregularly spaced i (e.g. lpc residuals defined along the path length)
  formed by applying an aggregating function to  of of a partitioned piecewise linear curve  from a 2 dimensional array with first column containing 
  '''
  def __init__(self):
    '''
    Constructor
    '''
    pass
  
class toyLPCPlotter():
  def __init__(self, evt):
    self._evt = evt
    self._fig = plt.figure(1)
    self._ax = Axes3D(self._fig)
    self._ax.set_title('toyTrack')
   
    self._points_state = False
    self._setColours()
    pass
  
  def setEvent(self, evt):
    self._evt = evt
 
  def clearEvent(self):
    plt.figure(self._fig.number)
    plt.cla()
    self._points_state = False
 
  def plotPoints(self):
    if not self._points_state:
      Xi = self._evt[0]['Xi']
      self._ax.scatter(Xi[:, 0], Xi[:, 1], Xi[:, 2], c = 'black', alpha = 0.1)
      self._fig.canvas.draw()
      self._points_state = True  
  
  def plotAllCurves(self, include_points = False):
    if include_points and not self._points_state:
      self.plotPoints()
    lpc_curves = self._evt[0]['lpc_curve']
    for k, col in enumerate(self._cols):
      self._plotCurve(lpc_curves[k], col)
    self._fig.canvas.draw()
  
  def plotCurve(self, k, include_points = False):
    if include_points and not self._points_state:
      self.plotPoints()
    lpc_curve = self._evt[0]['lpc_curve'][k]  
    self._plotCurve(lpc_curve, self._cols[k])    
    self._fig.canvas.draw()
    
  def _setColours(self):
    num_lpc_curves = len(self._evt[0]['lpc_curve'])
    self._cols = [col for i, col in zip(range(num_lpc_curves),cycle('bgrcmyk'))]
    
  def _plotCurve(self, lpc_curve, col):
    start_point = lpc_curve['start_point']
    self._ax.scatter([start_point[0]],[start_point[1]], [start_point[2]], c = col, marker = '^', s = [10])
    curve = lpc_curve['save_xd']
    self._ax.plot(curve[:,0],curve[:,1],curve[:,2], c = col, linewidth = 3, alpha = 0.5)
    #plt.figure(1)
    #plt.plot(lpc_curve[k]['lamb'], lpc_curve[k]['cos_neu_neu'], color = col)
    #plt.figure(2)
    #plt.plot(lpc_curve[k]['lamb'], lpc_curve[k]['rho'], color = col)
    
    
    