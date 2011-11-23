'''
Created on 6 Oct 2011

@author: droythorne
''' 
from copy import deepcopy
from lpcm.lpcStartPoints import lpcRandomStartPoints
from math import sqrt
from numpy.core.fromnumeric import prod, transpose
from numpy.core.numeric import array, ones, empty, zeros, dot
from numpy.core.shape_base import vstack, hstack
from numpy.lib.function_base import average
from numpy.lib.shape_base import vsplit
from numpy.linalg.linalg import eig, eigh
from numpy.ma.core import mean, mod, ravel, where, copy, sqrt
from scipy.spatial.distance import cdist
from scipy.stats.distributions import norm as gauss
from scitools.Lumpy import iscallable
from scitools.PrmDictBase import PrmDictBase
import numpy


class Direction:
  FORWARD = 1
  BACK = -1
  
class LPCImpl(PrmDictBase):
  '''
  Implements lpc calculation to mimic behaviour of lpc.R code and associated R functions/classes
  '''
  @staticmethod
  def _positivityCheck(x):
    return isinstance(x, (int, float)) and x > 0
  
  def _rescaleInput(self):
    scaled = self._lpcParameters['scaled']
    h = self._lpcParameters['h']
    
    if scaled: 
      self._dataRange = numpy.max(self.Xi, axis = 0) - numpy.min(self.Xi, axis = 0) #calculate ranges of each dimension
      self.Xi = self.Xi / self._dataRange
      if h is None:
        h = 0.1
        self.resetScaleParameters(h, self._lpcParameters['t0'])  
    else:
      if h is None:
        self._dataRange = numpy.max(self.Xi, axis = 0) - numpy.min(self.Xi, axis = 0) #calculate ranges of each dimension
        h = list(0.1 * self._dataRange)
        self.resetScaleParameters(h, self._lpcParameters['t0'])
    #make sure that t0 is set 
    if self._lpcParameters['t0'] is None:
      self._lpcParameters['t0'] = mean(h)
  def _selectStartPoints(self, x0):
    '''
    Delegates to _selectStartPoints the task of generating a number, mult, seed points for multiple passes of the algorithm 
    '''
    mult = self._lpcParameters['mult']
    
    self.set_in_dict('mult', mult, self._lpcParameters)
    
    self.x0 = self._startPointsGenerator(self.Xi, n = mult, x0 = x0)
  
  def _kern(self, y, x = 0, h = 1):
    return gauss.pdf((y-x)/h) / h
  def _kernd(self, X, x, h):
    return prod(self._kern(X, x, h), axis = 1)
  def _followx( self, x, way = 'one', phi = 1, last_eigenvector = None, weights = 1.):  
    if way == 'one':
      curve = self._followxSingleDirection(
                                  x, 
                                  direction = Direction.FORWARD,
                                  phi = phi,
                                  last_eigenvector = last_eigenvector,
                                  weights = weights)
      curve['start_point'] = x
      curve['start_point_index'] = 0
      return curve
    elif way == 'back':
      curve = self._followxSingleDirection(
                                  x,
                                  direction = Direction.BACK,
                                  phi = phi,
                                  last_eigenvector = last_eigenvector,
                                  weights = weights)
      curve['start_point'] = x
      curve['start_point_index'] = 0
      return curve
    elif way == 'two':
      forward_curve =  self._followxSingleDirection(
                                  x, 
                                  direction = Direction.FORWARD,
                                  phi = phi,
                                  last_eigenvector = last_eigenvector,
                                  weights = weights)
      back_curve =  self._followxSingleDirection(
                                  x,
                                  direction = Direction.BACK,
                                  phi = phi,
                                  last_eigenvector = last_eigenvector,
                                  weights = weights)
      #Stitching - append forward_curve to the end of the reversed back_curve with initial point of back curve removed
      #TODO - neaten this up, looks pretty clumsy
      combined_distance = hstack((-back_curve['lamb'][:0:-1], forward_curve['lamb'])) 
      curve = {'start_point': x,
               'start_point_index': len(back_curve['save_xd']) - 1,
               'save_xd': vstack((back_curve['save_xd'][:0:-1], forward_curve['save_xd'])),
               'eigen_vecd': vstack((back_curve['eigen_vecd'][:0:-1], forward_curve['eigen_vecd'])),
               'cos_neu_neu': hstack((back_curve['cos_neu_neu'][:0:-1], forward_curve['cos_neu_neu'])),
               'rho': hstack((back_curve['rho'][:0:-1], forward_curve['rho'])),
               'high_rho_points': vstack((back_curve['high_rho_points'], forward_curve['high_rho_points'])),
               'lamb': combined_distance - min(combined_distance),
               'c0': hstack((back_curve['c0'][:0:-1], forward_curve['c0'])),
               }      
      return curve
    else:
      raise ValueError, 'way must be one of one/back/two'
  def _calculatePathLength(self, save_xd):
    '''Calculates the cumulative Euclidian d-dimensional path length of the piecewise linear curve defined by a series of points save_xd
       Returns an array with n entries, where n is the number of points defining the curve. First point has distance 0.
    ''' 
    it = len(save_xd)
    lamb = empty(it)
    for i in range(it):
      if i==0:
        lamb[0] = 0
      else:
        lamb[i] = lamb[i-1] + sqrt(sum((save_xd[i] - save_xd[i-1])**2))
    return lamb
  
  def _followxSingleDirection(  self, 
                                x, 
                                direction = Direction.FORWARD,
                                forward_curve = None, 
                                phi = 1, 
                                last_eigenvector = None, 
                                weights = 1.):

    x0 = copy(x)
    N = self.Xi.shape[0]
    d = self.Xi.shape[1]
    it = self._lpcParameters['it']
    h = array(self._lpcParameters['h'])
    t0 = self._lpcParameters['t0']
    rho0 = self._lpcParameters['rho0']
    
    save_xd = empty((it,d))
    eigen_vecd = empty((it,d))
    c0 = ones(it)
    cos_alt_neu = ones(it)
    cos_neu_neu = ones(it)    
    lamb = empty(it) #NOTE this is named 'lambda' in the original R code
    rho = zeros(it)
    high_rho_points = empty((0,d))    
    
    count_points = 0
    
    for i in range(it):
      kernel_weights = self._kernd(self.Xi, x0, c0[i]*h) * weights
      mu_x = average(self.Xi, axis = 0, weights = kernel_weights)
      sum_weights = sum(kernel_weights)
      mean_sub = self.Xi - mu_x 
      cov_x = dot( dot(transpose(mean_sub), numpy.diag(kernel_weights)), mean_sub) / sum_weights 
     # assert (abs(cov_x.transpose() - cov_x)/abs(cov_x.transpose() + cov_x) < 1e-6).all(), 'Covariance matrix not symmetric, \n cov_x = {0}, mean_sub = {1}'.format(cov_x, mean_sub)
      save_xd[i] = mu_x #save first point of the branch
      count_points += 1
      #Calculate path length
      if i==0:
        lamb[0] = 0
      else:
        lamb[i] = lamb[i-1] + sqrt(sum((mu_x - save_xd[i-1])**2))
      
      #Calculate eigenvalues/vectors
      #(sorted_eigen_cov is a list of tuples containing eigenvalue and associated eigenvector, sorted descending by eigenvalue)
      eigen_cov = eigh(cov_x)
      sorted_eigen_cov = zip(eigen_cov[0],map(ravel,vsplit(eigen_cov[1].transpose(),len(eigen_cov[1]))))
      sorted_eigen_cov.sort(key = lambda elt: elt[0], reverse = True)   
      eigen_norm = sqrt(sum(sorted_eigen_cov[0][1]**2))
      eigen_vecd[i] = direction * sorted_eigen_cov[0][1] / eigen_norm  #Unit eigenvector corresponding to largest eigenvalue
      rho[i] = sorted_eigen_cov[1][0] / sorted_eigen_cov[0][0] #Ratio of two largest eigenvalues
      
      if i != 0 and rho[i] > rho0 and rho[i-1] <= rho0:
        high_rho_points = vstack((high_rho_points, x0))
      if i==0 and last_eigenvector is not None:
        cos_alt_neu[i] = direction * dot(last_eigenvector, eigen_vecd[i])
      if i > 0:
        cos_alt_neu[i] = dot(eigen_vecd[i], eigen_vecd[i-1])
      #Signum flipping
      if cos_alt_neu[i] < 0:
        eigen_vecd[i] = -eigen_vecd[i]
        cos_neu_neu[i] = -cos_alt_neu[i]
      else:
        cos_neu_neu[i] = cos_alt_neu[i]
      
      pen = self._lpcParameters['pen']
      if pen > 0:
        if i == 0 and last_eigenvector is not None:
          a = abs(cos_alt_neu[i])**pen
          eigen_vecd[i] = a * eigen_vecd[i] + (1-a) * last_eigenvector
        if i > 0:
          a = abs(cos_alt_neu[i])**pen
          eigen_vecd[i] = a * eigen_vecd[i] + (1-a) * eigen_vecd[i-1]
              
      #Check curve termination criteria
      if i not in (0, it-1):
        #Crossing
        cross = self._lpcParameters['cross']
        
        if forward_curve is None:
          full_curve_points = save_xd[0:i+1]
        else:
          full_curve_points = vstack((forward_curve['save_xd'],save_xd[0:i+1])) #inefficient, initialize then append? 
        
        if not cross:
          prox = where(ravel(cdist(full_curve_points,[mu_x])) <= mean(h))[0]
          if len(prox) != max(prox) - min(prox) + 1:
            break
          
        #Convergence
        convergence_at = self._lpcParameters['convergence_at']
        conv_ratio = abs(lamb[i] - lamb[i-1]) / (2 * (lamb[i] + lamb[i-1]))
        if conv_ratio  < convergence_at:
          break
        
        #Boundary
        boundary = self._lpcParameters['boundary']
        if conv_ratio < boundary:
          c0[i+1] = 0.995 * c0[i]
        else:
          c0[i+1] = min(1.01*c0[i], 1)
      
      #Step along in direction eigen_vecd[i]
      x0 = mu_x + t0 * eigen_vecd[i]
    
    #Trim output in the case where convergence occurs before 'it' iterations    
    curve = { 'save_xd': save_xd[0:count_points],
              'eigen_vecd': eigen_vecd[0:count_points],
              'cos_neu_neu': cos_neu_neu[0:count_points],
              'rho': rho[0:count_points],
              'high_rho_points': high_rho_points,
              'lamb': lamb[0:count_points],
              'c0': c0[0:count_points]
            }
    return curve  

  def __init__(self, start_points_generator = lpcRandomStartPoints(), **params):
    '''
    TODO document each parameter within dict and copy warnings/errors (possible?)
    
    '''
    super(LPCImpl, self).__init__()
    
    self._lpcParameters = { 'h': None, 
                            't0': None,
                            'way': 'two',
                            'scaled': True,
                            'pen': 2,
                            'depth': 1,
                            'it': 100,
                            'cross': True,
                            'boundary': 0.005,
                            'convergence_at': 1e-6,
                            'mult': None, #set this to None to allow exactly the number of local density modes to be returned from MeanShift  
                            'pruning_thresh': 0.0,
                            'rho0': 0.4,
                            'gapsize': 1.5  
                          }
    
    self._prm_list = [self._lpcParameters] 
    self.user_prm = None #extension of parameter set disallowed
    self._type_check.update({ 'h': lambda x: (x == None) or LPCImpl._positivityCheck(x) or (isinstance(x, list) and all(map(LPCImpl._positivityCheck, x)) ) , 
                              't0': lambda x: (x == None) or LPCImpl._positivityCheck,
                              'way': lambda x: x in ('one', 'two', 'back'),
                              'scaled': (bool,),
                              'pen': lambda x: (x == 0) or LPCImpl._positivityCheck,
                              'depth': lambda x: x in (1,2,3),
                              'it': lambda x: isinstance(x, int) and x > 9,
                              'cross': (bool,),
                              'convergence_at': LPCImpl._positivityCheck,
                              'boundary':  lambda x: LPCImpl._positivityCheck(x), #TODO - no assertion boundary > convergence.at (add to _update)
                              'mult': lambda x: (x == None) or (isinstance(x, int) and x > 0),
                              'pruning_thresh': LPCImpl._positivityCheck,
                              'rho0': LPCImpl._positivityCheck,
                              'gapsize': LPCImpl._positivityCheck
                            })
    self.set(**params)
    
    self.Xi = None
    self.x0 = None
    self._dataRange = None
    self._curve = None
   
    if not iscallable(start_points_generator):
      raise TypeError, 'Start points generator must be callable'
    self._startPointsGenerator = start_points_generator
    #self._startPointsGenerator.setScaleParameters(self._lpcParameters['h'])
    
    
  def getCurve(self, unscale = False):
    '''Returns a deep copy of self._curve, unless if unscale = True and self._lpcParameters['scaled'] = True, then a deep copy of self._curve is 
       returned, with save_xd, start_point and high_rho_points multiplied through by range parameters, self._dataRange. lamb is then recalculated on the unscaled data#
       All other elements of the lpc are unchanged
    '''
    curve = deepcopy(self._curve)
    
    if unscale == True and self._lpcParameters['scaled'] == True:
      for c in curve:
        c['start_point'] = c['start_point'] * self._dataRange
        c['save_xd'] = c['save_xd'] * self._dataRange
        c['high_rho_points'] = c['save_xd'] * self._dataRange
        c['lamb'] = self._calculatePathLength(c['save_xd'])
    
    return curve
    
  def resetScaleParameters(self, h, t0 = None):
    self.set_in_dict('h', h, self._lpcParameters)
    self._startPointsGenerator.setScaleParameters(self._lpcParameters['h'])
    if t0 is None:
      t0 = mean(h)  
    self.set_in_dict('t0', t0, self._lpcParameters)
    
  def setDataPoints(self, X):
    
    if X.ndim != 2:
      raise ValueError, 'X must be 2 dimensional'
    d = X.shape[1] 
    if d==1:
      raise ValueError, 'Data set must be at least two-dimensional'
    self.Xi = array(X, dtype = float)
    self._rescaleInput() #NOTE, scaling should take place prior to start points being generated
    print 'breakpoint'
  def lpc(self, x0 = None, X=None, weights = None):
    ''' Will return the scaled curve if self._lpcParameters['scaled'] = True, to return the curve on the same scale as the originally input data, call getCurve with unscale = True
    '''
    if X is None:
      if self.Xi is None:
        raise ValueError, 'Data points have not yet been set in this LPCImpl instance. Either supply as X parameter to this function or call setDataPoints'
    else:
      self.setDataPoints(X)
        
    N = self.Xi.shape[0]
    if weights is None:
      w = ones(N, dtype = float)
    else:
      w = array(weights, dtype = float)
      if w.shape != (N):
        raise ValueError, 'Weights must be one dimensional of vector of weights with size equal to the sample size'
    
    
    self._selectStartPoints(x0)
        
    #TODO add initialization relevant for other branches
    m = self.x0.shape[0] #how many starting points were actually generated
    way = self._lpcParameters['way']
    self._curve = [self._followx(self.x0[j], way = way, weights = w) for j in range(m)]
    return self._curve
      
      