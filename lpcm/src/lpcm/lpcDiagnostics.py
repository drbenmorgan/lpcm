'''
Created on 28 Oct 2011

@author: droythorne
'''
from numpy import isinf, invert, mean, std
from numpy.core.numeric import cross, dot, array
from numpy.core.shape_base import vstack, hstack
from numpy.ma.core import empty, zeros, sqrt, argsort
from scipy.spatial.kdtree import KDTree
from scitools.PrmDictBase import PrmDictBase
class LPCResiduals(PrmDictBase):
  '''
  classdocs
  '''
  _TUBE_BALL_MULTIPLICITY_FACTOR = 2
  
  def _calculateBallRadius(self, curve, tube_radius):
    if self._maxSegmentLength is None:
      self._maxSegmentLength = self._calculateMaxSegmentLength(curve)
    return max(0.5*self._maxSegmentLength, 1.1547 * tube_radius) 
  def _calculatePrunedPointResiduals(self, curve):
    ball_radius = self._calculateBallRadius(curve, self._params['tube_radius'])
    eps = self._params['eps']
    k = LPCResiduals._TUBE_BALL_MULTIPLICITY_FACTOR * self._params['k']
    return self._treeX.query(curve['save_xd'], k, eps, 2.0, ball_radius)
  def _calculateNNBallIndices(self, curve, ball_radius):
    ball_radius = self._calculateBallRadius(curve, ball_radius)
    eps = self._params['eps']
    return self._treeX.query_ball_point(curve['save_xd'], ball_radius, 2.0, eps)
  '''Returns tuple of minimum distance to the directed line segment AB from p, and the distance along AB of the point of intersection'''  
  def _distancePointToLineSegment(self, a, b, p):
    ab_mag2 = dot((b-a),(b-a))
    pa_mag2 = dot((a-p),(a-p))
    pb_mag2 = dot((b-p),(b-p))
    if pa_mag2 + ab_mag2 <= pb_mag2:
      return (sqrt(pa_mag2),0)
    elif pb_mag2 + ab_mag2 <= pa_mag2:
      return (sqrt(pb_mag2), sqrt(ab_mag2))
    else:
      c = cross((b-a),(p-a))
      if ab_mag2 == 0:
        raise ValueError, 'Division by zero magnitude line segment AB'
      dist_to_line2 = dot(c,c)/ab_mag2
      dist_to_line = sqrt(dist_to_line2)
      dist_along_segment = sqrt(pa_mag2 - dist_to_line2)
      return (dist_to_line, dist_along_segment)
  def _calculateMaxSegmentLength(self, curve):
    return max([curve['lamb'][i+1] - curve['lamb'][i] for i in range(len(curve['lamb']) - 1)])      
  def __init__(self, X, **params):
    '''
    Constructor
    '''
    self._treeX = KDTree(X)
    self._X = X #trat as read only
    self._maxSegmentLength = None
    super(LPCResiduals, self).__init__()
    self._params = {  'k': 20, 
                      'tube_radius': 0.2,
                      'eps': 0.0 
                   }
    self._prm_list = [self._params] 
    self.user_prm = None #extension of parameter set disallowed
    self._type_check.update({ 'k': lambda x: isinstance(x, int) and x > 0, 
                              'tube_radius': lambda x: isinstance(x, (int,float)) and x > 0,
                              'eps': lambda x: isinstance(x, (int,float)) and x > 0
                            })
    self.set(**params)
  
  '''Calculates the indices of self._X that contain points within tau of the curve defined by lpc_points'''
  def _calculateCoverageIndices(self, curve, tau):
    indices = self._calculateNNBallIndices(curve, self._calculateBallRadius(curve, tau))
    lpc_points = curve['save_xd']
    num_lpc_pts = len(lpc_points)
    points_in_tube = set()
    
    for i in range(num_lpc_pts - 1):
      trial_indices = list(set(indices[i:i+2].ravel()[0]))
      trial_points = self._X[trial_indices] 
      local_points_in_tube = []
      for j, p in enumerate(trial_points):
        d = self._distancePointToLineSegment(lpc_points[i], lpc_points[i+1], p)[0]
        if d < tau:
          local_points_in_tube.append(trial_indices[j])
      points_in_tube = points_in_tube | set(local_points_in_tube)  
    return points_in_tube
  '''Return a 2*(len(lpc_points) - 1) array of the proportion of self._X points within tau (in tau_range, an array)'''
  '''of the curve segments, where 'curve' is a element (dictionary) of curve returned by LPCImpl.lpc''' 
  def getCoverageGraph(self, curve, tau_range):
    coverage = [1.0*len(self._calculateCoverageIndices(curve,tau))/len(self._X) for tau in tau_range]
    return array([tau_range,coverage])
  '''This should give graphs similar to the output of BOakley'''
  def getGlobalResiduals(self, curve):
    if self._maxSegmentLength is None:
      self._maxSegmentLength = self._calculateMaxSegmentLength(curve)
    lpc_points = curve['save_xd']
    num_lpc_points = len(lpc_points)
    tree_lpc_points = KDTree(lpc_points)
    residuals = empty(len(self._X))
    residuals_lamb = empty(len(self._X))
    path_length = curve['lamb']
    
    for j, p in enumerate(self._X): 
      closest_lpc_point = tree_lpc_points.query(p)
      candidate_radius = sqrt(closest_lpc_point[0]**2 + 0.25*self._maxSegmentLength**2)
      candidate_segment_ends = tree_lpc_points.query_ball_point(p, candidate_radius)
      candidate_segment_ends.sort()
      
      current_min_segment_dist = (closest_lpc_point[0],0)
      current_closest_index = closest_lpc_point[1]
      last_index = None
      for i, index in enumerate(candidate_segment_ends):
        if index!=0 and last_index != index - 1:
          prv_segment_dist = self._distancePointToLineSegment(lpc_points[index-1], lpc_points[index], p)
          if prv_segment_dist[0] < current_min_segment_dist[0]:
            current_min_segment_dist = prv_segment_dist
            current_closest_index = index - 1
        if index !=  num_lpc_points - 1:  
          prv_segment_dist = self._distancePointToLineSegment(lpc_points[index], lpc_points[index+1], p)
          if prv_segment_dist[0] < current_min_segment_dist[0]:
            current_min_segment_dist = prv_segment_dist
            current_closest_index = index
        last_index = index
      residuals[j] = current_min_segment_dist[0]
      residuals_lamb[j] = path_length[current_closest_index] + current_min_segment_dist[1]
    lamb_order = argsort(residuals_lamb)
    return (residuals_lamb[lamb_order], residuals[lamb_order])
  
  def getPathResidualDiags(self, curve):
    lpc_points = curve['save_xd']
    residuals = self._calculatePrunedPointResiduals(curve)
    #strip inf values from arrays with less than k NNs within radius_threshold
    point_dist = residuals[0]
    point_dist = [point_dist[j][invert(isinf(point_dist[j]))] for j in range(point_dist.shape[0])]
    k = self._params['k']
    num_NN = map(len, point_dist[:k])
    mean_NN = map(mean,point_dist[:k])
    std_NN = map(std, point_dist[:k])
    #indices will contain entries equal to self._X.shape[0], which are out of bounds
    #these are removed with the set symm difference below
    indices = residuals[1]
    num_tree_pts = set([self._X.shape[0]])
    num_lpc_pts = len(lpc_points)
    line_seg_mean_NN = zeros(num_lpc_pts - 1)
    line_seg_std_NN = zeros(num_lpc_pts - 1)
    line_seg_num_NN = zeros(num_lpc_pts - 1)
    
    for i in range(num_lpc_pts - 1):
      trial_points = self._X[list(set(indices[i:i+2].ravel()) - num_tree_pts)]
      if len(trial_points) != 0:
        line_seg_NN_dists = empty(len(trial_points))
        j = 0 
        for p in trial_points:
          line_seg_NN_dists[j] = self._distancePointToLineSegment(lpc_points[i], lpc_points[i+1], p)[0]
          j = j + 1
        line_seg_NN_dists.sort()
        
        line_seg_num_NN[i] = min(len(line_seg_NN_dists), k)
        line_seg_mean_NN[i] = mean(line_seg_NN_dists[:k])
        line_seg_std_NN[i] = std(line_seg_NN_dists[:k])
      else:
        line_seg_num_NN[i] = 0
        line_seg_mean_NN[i] = 0.0
        line_seg_std_NN[i] = 0.0
      
    return {'num_NN': num_NN, 'mean_NN': mean_NN, 'std_NN': std_NN, 
            'line_seg_num_NN': line_seg_num_NN, 'line_seg_mean_NN': line_seg_mean_NN, 'line_seg_std_NN': line_seg_std_NN}
      