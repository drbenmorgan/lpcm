'''
Created on 6 Oct 2011

@author: droythorne

lpc
===

This module provides the depth 1, local principal curve algorithm as described 
Einbeck, J., Tutz, G., & Evers, L. (2005), Local principal curves, Statistics and Computing 15, 301-313
and has arguments, defaults and behaviour similar to that of the lpc function in CRAN package 'LPCM'.    
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
from numpy.ma.core import mean, mod, ravel, where, copy
from scipy.spatial.distance import cdist
from scipy.stats.distributions import norm as gauss
from scitools.Lumpy import iscallable
from scitools.PrmDictBase import PrmDictBase
import numpy

__all__ = ['Direction', 'LPCImpl']

class Direction:
  '''Defines the orientation of an lpc curve
  ''' 
  FORWARD = 1
  BACK = -1
  
class LPCImpl(PrmDictBase):
  '''Functor that implements lpc calculation to mimic behaviour of lpc.R code and associated R functions/classes. Computes the actual local
  principal curve, i.e. a sequence of local centers of mass, along with other sequences defined at the local centers of mass that characterise
  the curve (e.g. cumulative path length) and and its relationship with the data points (e.g. ratio of largest two principal component eigenvalues)
  '''
  @staticmethod
  def _positivityCheck(x):
    return isinstance(x, (int, float)) and x > 0
  
  def _selectStartPoints(self, x0):
    '''Delegates to _selectStartPoints the task of generating mult seed points for multiple passes of the lpc algorithm. Depending on the _startPointsGenerator
    used, x0 will either act as seed points for the start points generator (lpcMeanShift), or explicit start points (lpcRandomStartPoints)
    
    x0: 2-dim (n,m) numpy.array of floats, n is num of seed points, m is dim of feature space specifies the choice of starting points
      or seeds for starting points. The eventual number of starting points returned by self._startPointsGenerator is determined by mult - 
      if mult is an integer, it should return exactly mult start points, otherwise
          
          The default choice
          Optionally, one can also set one or more starting points manually here. This can
          be done in form of a matrix, where each row corresponds to a
          starting point, or in form of a vector, where starting points
          are read in consecutive order from the entries of the vector.
          The starting point has always to be specified on the original
          data scale, even if scaled=TRUE. A fixed number of starting
          points can be enforced through option mult in
          lpc.control.
    '''
    mult = self._lpcParameters['mult']
    self.x0 = self._startPointsGenerator(self.Xi, n = mult, x0 = x0)
  
  def _kern(self, y, x = 0.0, h = 1.0):
    '''Gaussian kernel, gives the weight ascribed to points in y relative to x with scale parameter h
    
    Parameters
    ----------
    y : 1-dim length m, or 2 dimensional (n*m) numpy.array of floats containing coordinates of n, m-dimensional feature points 
    x : float or 1-dim length m numpy.array of floats containing the coordinates of point relative to which kernel weights are calculated
    h : float or 1-dim length m numpy.array of floats containing the scale parameter for each dimension
    NOTE - the final division through by h is present in the original R package, but I'm fairly sure it's an error  
    '''
    return gauss.pdf((y-x)/h) / h
  
  def _kernd(self, X, x, h):
    '''Computes separable Gaussian kernel (product of m 1-dim kernels for each coordinate) for each of the n, m-dim feature points of X 
    relative to point x with scale parameters h (see _kern parameters)
    
    Returns
    -------
    w: 1-dim, length n array of weights of feature points X relative to x
    '''
    w = prod(self._kern(X, x, h), axis = 1)  
    return w
  
  def _followx( self, x, way = 'one', last_eigenvector = None, weights = 1.):
    '''Generates a single lpc curve, from the start point, x. Proceeds in forward ('one'), backward ('back') or both ('two') 
    directions from this point.  
    
    Parameters
    ----------
    x : 1-dim numpy.array of floats containing the start point for the lpc algorithm
    way : one of 'one'/'back'/'two', defines the orientation of the lpc propagation
    last_eigenvector: see _followXSingleDirection
    weights: see _followXSingleDirection
    
    Returns
    -------
    curve : a dictionary comprising a single lpc curve in m-dim feature space, with keys, values as self._followxSingleDirection
    with the addition of
              
      start_point, 1-dim numpy.array of floats of length m;
      start_point_index, index of start_point in save_xd;
    
      For way == 'two', the forward and backward curves are stitched together. save_xd, eigen_vecd, cos_neu_neu, rho and c0 are formed
      by concatenating the reversed 'back' curve (with start_point removed) with the 'one' curve. high_rho_points are the union of 
      forward and backward high_rho_points. lamb is the cumulative segment distance along the stitched together save_xd with, as before,
      lamb[0] = 0.0. TODO, should farm this out to an 'lpcCurve'-type class that knows how to join its instances          
    '''
    if way == 'one':
      curve = self._followxSingleDirection(
                                  x, 
                                  direction = Direction.FORWARD,
                                  last_eigenvector = last_eigenvector,
                                  weights = weights)
      curve['start_point'] = x
      curve['start_point_index'] = 0
      return curve
    elif way == 'back':
      curve = self._followxSingleDirection(
                                  x,
                                  direction = Direction.BACK,
                                  last_eigenvector = last_eigenvector,
                                  weights = weights)
      curve['start_point'] = x
      curve['start_point_index'] = 0
      return curve
    elif way == 'two':
      forward_curve =  self._followxSingleDirection(
                                  x, 
                                  direction = Direction.FORWARD,
                                  last_eigenvector = last_eigenvector,
                                  weights = weights)
      back_curve =  self._followxSingleDirection(
                                  x,
                                  direction = Direction.BACK,
                                  forward_curve = forward_curve,
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
    TODO - factor this out into an 'lpcPath'-type class
    Parameters
    ----------
    save_xd : 2-dim (n*m) numpy.array of floats containing coordinates of n, ordered, m-dimensional feature points defining a
    piecewise linear curve with n-1 segments
    
    Returns
    ------- 
    lamb : 1-dim array with n ordered entries, defining the cumulative sum of segment lengths. lamb[0] = 0.
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
                                last_eigenvector = None, 
                                weights = 1.):
    '''Generates a partial lpc curve dictionary from the start point, x.
    Arguments
    ---------
    x : 1-dim, length m, numpy.array of floats, start point for the algorithm when m is dimension of feature space
    
    direction :  bool, proceeds in Direction.FORWARD or Direction.BACKWARD from this point (just sets sign for first eigenvalue) 
    
    forward_curve : dictionary as returned by this function, is used to detect crossing of the curve under construction with a
        previously constructed curve
        
    last_eigenvector : 1-dim, length m, numpy.array of floats, a unit vector that defines the initial direction, relative to
        which the first eigenvector is biased and initial cos_neu_neu is calculated  
        
    weights : 1-dim, length n numpy.array of observation weights (can also be used to exclude
        individual observations from the computation by setting their weight to zero.),
        where n is the number of feature points 
    '''
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
      #assert (abs(cov_x.transpose() - cov_x)/abs(cov_x.transpose() + cov_x) < 1e-6).all(), 'Covariance matrix not symmetric, \n cov_x = {0}, mean_sub = {1}'.format(cov_x, mean_sub)
      save_xd[i] = mu_x #save first point of the branch
      count_points += 1
      
      #calculate path length
      if i==0:
        lamb[0] = 0
      else:
        lamb[i] = lamb[i-1] + sqrt(sum((mu_x - save_xd[i-1])**2))
      
      #calculate eigenvalues/vectors
      #(sorted_eigen_cov is a list of tuples containing eigenvalue and associated eigenvector, sorted descending by eigenvalue)
      eigen_cov = eigh(cov_x)
      sorted_eigen_cov = zip(eigen_cov[0],map(ravel,vsplit(eigen_cov[1].transpose(),len(eigen_cov[1]))))
      sorted_eigen_cov.sort(key = lambda elt: elt[0], reverse = True)   
      eigen_norm = sqrt(sum(sorted_eigen_cov[0][1]**2))
      eigen_vecd[i] = direction * sorted_eigen_cov[0][1] / eigen_norm  #Unit eigenvector corresponding to largest eigenvalue
      
      #rho parameters
      rho[i] = sorted_eigen_cov[1][0] / sorted_eigen_cov[0][0] #Ratio of two largest eigenvalues
      if i != 0 and rho[i] > rho0 and rho[i-1] <= rho0:
        high_rho_points = vstack((high_rho_points, x0))
      
      #angle between successive eigenvectors
      if i==0 and last_eigenvector is not None:
        cos_alt_neu[i] = direction * dot(last_eigenvector, eigen_vecd[i])
      if i > 0:
        cos_alt_neu[i] = dot(eigen_vecd[i], eigen_vecd[i-1])
      
      #signum flipping
      if cos_alt_neu[i] < 0:
        eigen_vecd[i] = -eigen_vecd[i]
        cos_neu_neu[i] = -cos_alt_neu[i]
      else:
        cos_neu_neu[i] = cos_alt_neu[i]
     
      #angle penalization
      pen = self._lpcParameters['pen']
      if pen > 0:
        if i == 0 and last_eigenvector is not None:
          a = abs(cos_alt_neu[i])**pen
          eigen_vecd[i] = a * eigen_vecd[i] + (1-a) * last_eigenvector
        if i > 0:
          a = abs(cos_alt_neu[i])**pen
          eigen_vecd[i] = a * eigen_vecd[i] + (1-a) * eigen_vecd[i-1]
              
      #check curve termination criteria
      if i not in (0, it-1):
        #crossing
        cross = self._lpcParameters['cross']
        if forward_curve is None:
          full_curve_points = save_xd[0:i+1]
        else:
          full_curve_points = vstack((forward_curve['save_xd'],save_xd[0:i+1])) #inefficient, initialize then append? 
        if not cross:
          prox = where(ravel(cdist(full_curve_points,[mu_x])) <= mean(h))[0]
          if len(prox) != max(prox) - min(prox) + 1:
            break
          
        #convergence
        convergence_at = self._lpcParameters['convergence_at']
        conv_ratio = abs(lamb[i] - lamb[i-1]) / (2 * (lamb[i] + lamb[i-1]))
        if conv_ratio  < convergence_at:
          break
        
        #boundary
        boundary = self._lpcParameters['boundary']
        if conv_ratio < boundary:
          c0[i+1] = 0.995 * c0[i]
        else:
          c0[i+1] = min(1.01*c0[i], 1)
      
      #step along in direction eigen_vecd[i]
      x0 = mu_x + t0 * eigen_vecd[i]
    
    #trim output in the case where convergence occurs before 'it' iterations    
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
    Arguments
    ---------
    start_points_generator: a type callable with a single argument, x0, also implements the function setScaleParameters(h) 
    (see doc for function self.resetScaleParameters)

    h : 1-dim numpy.array, length m or float, bandwidth. May be either specified as a single number, then
        the same bandwidth is used in all dimensions, or as an
        m-dimensional bandwidth vector (where m is the dimension of feature points). 
        The default setting is 10 percent of the range in each direction. If scaled =TRUE
        then, if specified, the bandwidth has to be specified in fractions of the
        data range, e.g. h = [0.2, 0.1, 0.3], rather than absolute values.

    t0 : float, scalar step length. Default setting is 't0 = h' if 'h' is a
        scalar, and 't0 = mean(h)' if 'h' is a vector.

    depth (NOT USED): int, maximum depth of branches,  restricted to the
        values 1,2 or 3 (The original LPC branch has depth 1.  If,
        along this curve, a point features a high second local PC,
        this launches a new starting point, and the resulting branch
        has depth 2.  If, along this branch, a point features a high
        second local PC, this launches a new starting point, and the
        resulting branch has depth 3. )
 
    way: "one": go only in direction of the first local eigenvector,
        "back": go only in opposite direction, "two": go from
        starting point in both directions.

    scaled: bool, if True, scales each variable by dividing through its range
      
    pen: power used for angle penalization. If set to 0, the
        angle penalization is switched off.
          
    it : maximum number of iterations on either side of the starting
        point within each branch.
    
    cross:  If True, curves are stopped when they come
        too close to an existing branch. Used in the self-coverage
        function.
    
    boundary: This boundary correction [2] reduces the bandwidth adaptively
        once the relative difference of parameter values between two
        centers of mass falls below the given threshold. This measure
        delays convergence and enables the curve to proceed further
        into the end points. If set to 0, this boundary correction is
        switched off.
  
    convergence_at: this forces the curve to stop if the relative
        difference of parameter values between two centers of mass
        falls below the given threshold.  If set to 0, then the curve
        will always stop after exactly "iter" iterations.
  
    mult: integer which enforces a fixed number of starting
        points.  If the number given here is larger than the number
        of starting points provided at "x0", then the missing points
        will be set at random (For example, if d=2, mult=3, and
        x0=c(58.5, 17.8, 80,20), then one gets the starting points
        (58.5, 17.8), (80,20), and a randomly chosen third one.
        Another example for such a situation is "x0=NULL" with
        "mult=1", in which one random starting point is chosen). If
        the number given here is smaller the number of starting
        points provided at x0, then only the first "mult" starting
        points will be used.
    
    pruning_thresh (NOT USED) : float, used to remove non-dense, depth > 1 branches 
    
    rho0: float, steers the birth process of higher-depth starting points
        by acting as a threshold for definition of high_rho_pts. 
        Usually, between 0.3 and 0.4
                
    gapsize (NOT USED): float, sets scaling of t0 which is applied when 
        starting depth > 1 branches
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
    self.x0 = None #set equal to the return value of _selectStartPoints
    self._dataRange = None
    self._curve = None
    if not iscallable(start_points_generator):
      raise TypeError, 'Start points generator must be callable'
    self._startPointsGenerator = start_points_generator

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
    '''Sets the bandwidth as h and lpc segment length as t0. If t0 is None, t0 is set as mean(h). The scale 
    parameter for the start points generator is also set to h.
    Parameters
    ----------
    h : 1-dim, length m numpy.array or float where m is the dimension of the feature space
    t0 : float
    '''
    self.set_in_dict('h', h, self._lpcParameters)
    self._startPointsGenerator.setScaleParameters(self._lpcParameters['h'])
    if t0 is None:
      t0 = mean(h)  
    self.set_in_dict('t0', t0, self._lpcParameters)
  
  def _rescaleInput(self):
    '''If scaled == True : sets self._dataRange equal to a 1-dim numpy.array containing the ranges in each of m 
    feature space dimensions, and divides self.Xi through by these ranges (leaving the data in a 1*1*1 cube); 
    if the bandwidth, h, is not set, this defaults to 0.1 (i.e. 10% of the range). If scaled == False and h is not set, 
    h defaults to 10% of the range in each feature space dimension. In both the scaled and not scaled cases,
    if the segment length, t0, is not set this defaults to mean(h).
    '''
    scaled = self._lpcParameters['scaled']
    h = self._lpcParameters['h']
    if scaled: 
      data_range = numpy.max(self.Xi, axis = 0) - numpy.min(self.Xi, axis = 0) #calculate ranges of each dimension
      if any(data_range == 0):
        raise ValueError, 'Data cannot be scale because the range in at least 1 direction is zero (i.e. data lies wholly in plane x/y/z = c)'
      self._dataRange = data_range
      self.Xi = self.Xi / self._dataRange
      if h is None:
        self.resetScaleParameters(0.1, self._lpcParameters['t0'])  
    else:
      if h is None:
        self._dataRange = numpy.max(self.Xi, axis = 0) - numpy.min(self.Xi, axis = 0) #calculate ranges of each dimension
        h = list(0.1 * self._dataRange)
        self.resetScaleParameters(h, self._lpcParameters['t0'])
    #make sure that t0 is set 
    if self._lpcParameters['t0'] is None:
      self._lpcParameters['t0'] = mean(h)  
  
  def setDataPoints(self, X):
    '''Set the data points self.Xi as X, rescale and adjust bandwidth (h)/segment length (t0) if necessary
    ''' 
    if X.ndim != 2:
      raise ValueError, 'X must be 2 dimensional'
    d = X.shape[1] 
    if d==1:
      raise ValueError, 'Data set must be at least two-dimensional'
    self.Xi = array(X, dtype = float)
    self._rescaleInput() #NOTE, scaling should take place prior to start points being generated

  def lpc(self, x0 = None, X=None, weights = None):
    ''' Will return the scaled curve if self._lpcParameters['scaled'] = True, to return the curve on the same scale as the originally input data, call getCurve with unscale = True
    Arguments
    ---------
    x0 : 2-dim numpy.array containing #rows equal to number of explicitly defined start points
    and #columns equal to dimension of the feature space points; seeds for the start points algorithm
    X : 2-dim numpy.array containing #rows equal to number of data points and #columns equal to dimension 
    of the feature space points   
    weights : see self._followxSingleDirection docs
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
      
      