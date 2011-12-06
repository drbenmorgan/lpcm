'''
Created on 22 Nov 2011

@author: droythorne
'''
from itertools import cycle
from mpl_toolkits.mplot3d import axes3d, Axes3D
import matplotlib.pyplot as plt

class lpcDynamicBinnedResiduals():
  '''
  EXPERIMENTAL
  Generates a 1 dimensional histogram from two  1-dimensional, length n arrays, where values in the the second array
  are a function of those in the first (which are irregularly spaced i (e.g. lpc residuals defined along the path length)
  formed by applying an aggregating function to  of of a partitioned piecewise linear curve  from a 2 dimensional array
  with first column containing ...
  '''
  def __init__(self, x, y, min_count, min_width):
    self.x = x
    self.y = y
    self.min_count = min_count
    self.min_width = min_width
    self.bin_boundaries = None
  def bin(self):
    self.bin_boundaries = []
    bin_count = 0
    if len(self.x):
      bin_min = self.x[0]
      self.bin_boundaries.append(bin_min)
    for x in self.x:
      if bin_count < self.min_count or x - bin_min <= self.min_width:
        bin_count += 1
      else:
        self.bin_boundaries.append(x)
        bin_min = x
        bin_count = 1  
  def agg(self, f):
    if self.bin_boundaries is None:
      self.bin()
    bin_aggregate = []
    temp_bin_bound_index = 1
    num_bins = len(self.bin_boundaries)
    bin_contents = []
    for i, x in enumerate(self.x):
      
      if num_bins == temp_bin_bound_index:
        bin_aggregate.append((self.bin_boundaries[temp_bin_bound_index - 1], f(self.y[i:])))        
        break
      else:                    
        if x  < self.bin_boundaries[temp_bin_bound_index]:
          bin_contents.append(self.y[i])
        else:
          bin_aggregate.append((self.bin_boundaries[temp_bin_bound_index - 1], f(bin_contents)))
          bin_contents = [self.y[i]]
          temp_bin_bound_index += 1
          if i == len(self.x) - 1:
            bin_aggregate.append((self.bin_boundaries[temp_bin_bound_index - 1], f(self.y[i:])))
    return bin_aggregate

class lpcEventPlotter():
  '''
  >>> a = ana.lpcAnalysisPickleReader('/tmp/muon-proton_1266297863573875304_meta.pkl')
  >>> evt = a.getEvent()
  >>> p = lpcplot.lpcEventPlotter(evt)
  >>> p.plotCurve([2])
  >>> evt = a.getEvent()
  >>> p.setEvent(evt)
  >>> p.plotCurve([1])
  '''
  def __init__(self, evt):
    self._evt = evt
    self._fig = plt.figure()
    self._ax = Axes3D(self._fig)
    self._ax.set_title('toyTrack')
    self._points_state = False
    self._setColours()
    self.setCurvePlotter(lpcCurvePlotter(self._evt))
     
  def setCurvePlotter(self, curve_plotter):
    self._curvePlotter = curve_plotter
    self._curvePlotter.setColours(self.getColours())
  
  def setEvent(self, evt):
    self._evt = evt
    self._setColours()
    self.clearEvent()
  
  def clearEvent(self):
    plt.figure(self._fig.number)
    plt.cla()
    self._ax = Axes3D(self._fig)
    self._ax.set_title('toyTrack')
    self._points_state = False
    self._curvePlotter.clear()
  
  def getColours(self):
    '''Needed when associating colours of plots in other curve plotting classes with the event display
    '''
    return self._cols
  
  def plotPoints(self):
    if not self._points_state:
      Xi = self._evt[0]['Xi']
      self._ax.scatter(Xi[:, 0], Xi[:, 1], Xi[:, 2], c = 'black', alpha = 0.1)
      self._fig.canvas.draw()
      self._points_state = True
  
  def plotAllCurves(self, include_points = False):
    plt.figure(self._fig.number) 
    if include_points and not self._points_state:
      self.plotPoints()
    lpc_curves = self._evt[0]['lpc_curve']
    for k, col in enumerate(self._cols):
      self._plotCurve(lpc_curves[k], col)
    self._fig.canvas.draw() 
    self._curvePlotter.plot()
  
  def plotCurve(self, curve_indices, include_points = False):
    plt.figure(self._fig.number) 
    if include_points and not self._points_state:
      self.plotPoints()
    for i in curve_indices: 
      lpc_curve = self._evt[0]['lpc_curve'][i]  
      self._plotCurve(lpc_curve, self._cols[i])    
    self._curvePlotter.plot(curve_indices)
    self._fig.canvas.draw()
    
  def _setColours(self):
    num_lpc_curves = len(self._evt[0]['lpc_curve'])
    self._cols = [col for i, col in zip(range(num_lpc_curves),cycle('bgrcmyk'))]
    
  def _plotCurve(self, lpc_curve, col):
    start_point = lpc_curve['start_point']
    start_point_index = lpc_curve['start_point_index']
    direction = lpc_curve['eigen_vecd'][start_point_index]
    self._ax.scatter([start_point[0]],[start_point[1]], [start_point[2]], c = col, marker = 'o', s = [10])
    curve = lpc_curve['save_xd']
    self._ax.plot(curve[:,0],curve[:,1],curve[:,2], c = col, linewidth = 2, alpha = 0.2)
    self._ax.plot(curve[start_point_index:start_point_index + 5,0],  
                  curve[start_point_index:start_point_index + 5,1],
                  curve[start_point_index:start_point_index + 5,2],
                c = 'black', linewidth = 3, alpha = 0.7)
    self._ax.scatter( curve[start_point_index:start_point_index + 1,0],
                      curve[start_point_index:start_point_index + 1,1],
                      curve[start_point_index:start_point_index + 1,2], 
                      c = col, marker = '^')
    
class lpcCurvePlotter(object):
  '''Simple functor that takes any number of lpc curves and plots their cos_neu_neu/rho against lambda,
  marks points of high rho (perhaps add this to lpcEvent Plotter too?), marks the start point on each plot
  TODO - get this to take a residuals object and know how to plot these too
  '''
  def __init__(self, evt):
    self._evt = evt
    self._fig = plt.figure()
    self._rho = self._fig.add_subplot(211)
    self._cosneuneu = self._fig.add_subplot(212)
    self._setTitles()
    self._setDefaultColours()
  
  def _setTitles(self):
    self._rho.set_title('rho')
    self._cosneuneu.set_title('cos_neu_neu')
  
  def _setDefaultColours(self):
    num_lpc_curves = len(self._evt[0]['lpc_curve'])
    self._cols = list(num_lpc_curves * 'k')
  
  def setColours(self, cols):
    if len(cols) != len(self._evt[0]['lpc_curve']):
      raise ValueError, 'The number of colours must equal the number of lpc curves in the event'
    else:
      self._cols = cols
  
  def clear(self):
    self._rho.clear()
    self._cosneuneu.clear()
    self._fig.canvas.draw()

  def plot(self, curve_indices = None):
    '''Plots lpc curve properties in self._evt with indices equal element of list 'curve_indices'
    '''   
    plt.figure(self._fig.number) 
    self._setTitles()
    curves = self._evt[0]['lpc_curve']
    if curve_indices is None:
      curve_indices = range(len(curves))
    for c in curve_indices:
      self._rho.plot(curves[c]['lamb'], curves[c]['rho'], color = self._cols[c])
      self._cosneuneu.plot(curves[c]['lamb'], curves[c]['cos_neu_neu'], color = self._cols[c])
    start_lambda = [curves[c]['lamb'][curves[c]['start_point_index']] for c in curve_indices]
    plt.subplot(211)
    ymin, ymax = plt.ylim()
    for c, l in zip(curve_indices, start_lambda):
      self._rho.plot([l, l], [ymin, ymax], '--', color = self._cols[c])
    plt.subplot(212)
    ymin, ymax = plt.ylim()
    for c, l in zip(curve_indices, start_lambda):
      self._cosneuneu.plot([l,l], [ymin, ymax], '--', color = self._cols[c])
    self._fig.canvas.draw()
class lpcResidualsPlotter(object):
  pass   