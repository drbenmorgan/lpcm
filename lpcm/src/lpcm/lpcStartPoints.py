'''
Created on 15 Nov 2011

@author: droythorne
'''
from numpy.core.shape_base import vstack
from numpy.ma.core import mean, floor
from numpy.random import random_integers
from random import sample
from scitools.PrmDictBase import PrmDictBase
from sklearn.cluster.mean_shift_ import MeanShift

def lpcUserDefinedStartPoints(x0):
  '''Trivial identity function to forward user defined start points to lpc algorithm'''
  return x0
  
class lpcRandomStartPoints():
  '''
  Callable that selects points selected uniformally (with replacement) from the input points, X. 
  If explicitly defined starting points, x0, are passed when called, then if the required number of points, n, is less than 
  num of starting points, them function will take a random selection of n points from x0; otherwise the defined x0 will be topped up 
  to n with random selections (with replacement) from self.Xi.
  '''
  def __init__(self, X):
    '''
    X, 2-dimensional array containing #rows equal to number of data points and #columns equal to dimension of the data points
    '''
    self.Xi = X
  def __call__(self, n = 10, x0 = None):
    '''
    n, required number of start points, if  None, defaults ot 10 start points
    x0, 2-dimensional array containing #rows equal to number of explicitly defined start points and #columns equal to dimension of the data points
    '''
    if n is None or n == 0:
      n = 10
    if x0 is not None:
      num_x0_pts = self.x0.shape[0]
    else:
      return self.Xi[random_integers(0, self.Xi.shape[0] - 1, n),:]
    if num_x0_pts == n:
      return x0
    elif num_x0_pts < n:
      return vstack((x0,self.Xi[random_integers(0, self.Xi.shape[0] - 1, n - num_x0_pts),:]))
    else: #num_x0_pts > n
      return x0[sample(xrange(0,num_x0_pts), n),:]
    
class lpcMeanShift(PrmDictBase):
  '''
  Wrapper around the scikit-learn class sklearn.cluster.MeanShift to approximately mimic the behavior of the LPCM CRAN package
  Callable that generates n starting points based on local density modes of X. Seed points for modes are given by x0;
  ms.h controls the kernel bandwidth (original CRAN package allowed x0 as a vector with separate bandwidth per dimension,
  the sklearn MeanShift class allows only a scalar bandwidth, so the mean is taken), ms.sub is the percentage of data points 
  that should be used for selecting local density modes
'''
  @staticmethod
  def _positivityCheck(x):
    return isinstance(x, (int, float)) and x > 0
  
  def __init__(self, X, **params):
    '''
    X, 2 dimensional [#points, #dimension of points] array containing the data for which local density modes is to calculated 
    '''
    super(lpcMeanShift, self).__init__()
    self._lpcParameters = { 'ms_h': 0.1, 
                            'ms_sub': 30
                          }
    
    self._prm_list = [self._lpcParameters] 
    self.user_prm = None #extension of parameter set disallowed
    self._type_check.update({ 'ms_h': lambda x: lpcMeanShift._positivityCheck(x) or (isinstance(x, list) and all(map(lpcMeanShift._positivityCheck, x)) ) , 
                              'ms_sub': lambda x: lpcMeanShift._positivityCheck and x < 100
                           })
    self.set(**params)
    self._Xi = X
    
    bandwidth = mean(self._lpcParameters['ms_h'])
    self._meanShift = MeanShift(bandwidth = bandwidth)
    
    '''
    Generates n seed points for the lpc algorithm. 
    n, required number of seed points, if n = None, returns exactly the local density modes, otherwise lpcRandomStartPoints is called with x0 equal
    to the local density modes (local density modes are the cluster centers)
    x0, 2-dimensional array containing #rows equal to number of explicitly defined mean shift seed points and #columns equal 
    to dimension of the individual data points (called number fo features in MeanShift docs.
    
    Returns the lpc seed points as a 2 dimensional [#seed points, #dimension of seed points] array
    '''
  def __call__(self, n = None, x0 = None):
    if x0 is None:
      N = self._Xi.shape[0]
      ms_sub = float(self._lpcParameters['ms_sub'])
      #guarantees ms_sub <= ms_sub % of N <= 10 * ms_sub seed points (could give the option of using seed point binning in MeanShift)
      Nsub = int(min(max(ms_sub, floor(ms_sub * N / 100)), 10 * ms_sub))
      ms_seeds = self._Xi[sample(xrange(0,N), Nsub),:]
    else:
      ms_seeds = x0
    self._meanShift.seeds = ms_seeds
    self._meanShift.fit(self._Xi)
    
    if n is None:
      return self._meanShift.cluster_centers_
    else:
      lpcRSP = lpcRandomStartPoints(self._Xi)
      return lpcRSP(n, self._meanShift.cluster_centers_)
      
  def getClusterLabels(self):
    return self._meanShift.labels_
    
    