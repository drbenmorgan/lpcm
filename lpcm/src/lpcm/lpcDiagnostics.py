'''
Created on 28 Oct 2011

@author: droythorne
'''
from numpy import isinf, invert, mean, std
from numpy.core.numeric import cross, dot, array
from numpy.core.shape_base import vstack, hstack
from numpy.ma.core import empty, zeros, sqrt, argsort, ones
from scipy.spatial.kdtree import KDTree
from scitools.PrmDictBase import PrmDictBase

class LPCResidualsRunner():
  '''
  Generates curve diagnostics from an instance of LPCResiduals and an instance of LPCImpl These include pathResidualDiags,
  TubeResiduals and CoverageIndices for each curve, plus the matrix with element (n,m) equal to the proportion of coverage indices
  in cylinder m contained in cylinder n (i.e. the degree of overlap in the hits associated to each path, used later to prune the result set
  by removing or concatenating curves 
  '''
  def __init__(self, lpc_curves, lpc_residuals):
    self._lpcCurves = lpc_curves
    self._lpcResiduals = lpc_residuals
    self._tauRange = None
  def setTauRange(self,tau_range):
    '''
    tau_range, 1d list of floats , each defining the radius of the cylinder around lpc curves used to associate self._lpcAlgorithm.Xi points to curves
    '''   
    self._tauRange = tau_range
  def calculateResiduals(self):
    if self._tauRange is None:
      raise ValueError, 'tauRange, the list of cylinder radii, has not yet been defined'
    curves = self._lpcCurves
    curve_residuals = []
    '''
    for curve in curves:
        tube_residuals = self._lpcResiduals.getTubeResiduals(curve)
        path_residuals =  self._lpcResiduals.getPathResidualDiags(curve)
        coverage_indices = {}
        for tau in self._tauRange:
          indices = self._lpcResiduals.calculateCoverageIndices(curve, tau)
          coverage_indices[tau] = indices
        curve_residuals.append({'tube_residuals': tube_residuals, 'path_residuals': path_residuals, 'coverage_indices': coverage_indices})
    
    containment_matrices = {}
    for tau in self._tauRange:
      containment_matrix = self._calculateHitContainmentMatrix(curve_residuals, tau)
      containment_matrices[tau] = containment_matrix
    '''
    distance_matrix = self._calculateCurveDistanceMatrix(curves)
    residuals = {'distance_matrix': distance_matrix}
    #residuals = {'curve_residuals': curve_residuals, 'distance_matrix': distance_matrix, 'containment_matrices': containment_matrices}
    return residuals    
  
  def _calculateCurveDistanceMatrix(self, curves):
    num_curves = len(curves)
    distance_matrix = zeros((num_curves, num_curves))
    for i in range(num_curves):
      curve_i = curves[i]
      for j in range(i+1, num_curves):  
        curve_j = curves[j]
        distance_matrix[i,j] = self._lpcResiduals._distanceBetweenCurves(curve_i, curve_j)
        distance_matrix[j,i] = self._lpcResiduals._distanceBetweenCurves(curve_j, curve_i)
    return distance_matrix
  
  def _calculateHitContainmentMatrix(self, curve_residuals, tau):
    '''
    Calculates a 2d array where the (i,j)th entry is the proportion of elements in curve_residuals[i]['coverage_indices'][tau] that are also contained
    in curve_residuals[j]['coverage_indices'][tau] A value of -1 indicates that the curve corresponding to the row index had no hits within tau of the 
    curve
    ''' 
    num_curves = len(curve_residuals)
    containment_matrix = ones((num_curves, num_curves))
    for i in range(num_curves):
      labels_i = curve_residuals[i]['coverage_indices'][tau]
      for j in range(i+1, num_curves):  
        labels_j = curve_residuals[j]['coverage_indices'][tau]
        cardinality_intersect = len(labels_i & labels_j)
        if len(labels_i) == 0:
          containment_matrix[i, j] = -1
        else: 
          containment_matrix[i, j] = float(cardinality_intersect)/len(labels_i)
        if len(labels_j) == 0:
          containment_matrix[j,i] = -1
        else: 
          containment_matrix[j,i] = float(cardinality_intersect)/len(labels_j)
    return containment_matrix
    
class LPCResiduals(PrmDictBase):
  '''
  classdocs
  '''
  _TUBE_BALL_MULTIPLICITY_FACTOR = 2
  _CURVE_DIST_MAX_SEGMENTS = 50
  
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
  
  def _distanceBetweenCurves(self, l1, l2):
    '''Defines the asymmetric distance between two pielcewise linear curves, each defined by l1['save_xd']/l2['save_xd'] (a 2d (#points, #Euclidean dimension of points) array),
    as the average distance between the points defining l1 and the curve l2, where this distance between point and curve is the minimum value of 
    self._distancePointToLineSegment for line segments in l2
    Currently, since lines are at most iter segments long (as determined by lpc initialisation) and num iterations is typically far less than 500 since
    convergence happens long before iter iterations, the current code just scans all segments in l2 to find the minimum distance
    Works from endpoints rather than midpoints as in practice this should make only a nominal difference to the results
    This is likely to be a massive bottle neck as it scales with num of segments^2, so there's a hack included so that each curve is reduced to 
    a maximum number of segments by taking every other skip_l1 = (len(l1)-2) / max_segments elements from each curve (i.e calculate the distance from
    every skip_l1'th point of l1 to the segments determined by every skip_l2'th element of l2. max_segments must be an integer > 0
    '''
    weighted_dist = 0
    l1_curve = l1['save_xd']
    l1_lamb = l1['lamb']
    l2_curve = l2['save_xd']
    if len(l2_curve) < 2 or len(l1_curve) < 2:
        raise ValueError, 'Curves must contain at least 2 points'
  
    max_segments = LPCResiduals._CURVE_DIST_MAX_SEGMENTS
    if max_segments is None:
      skip_l1 = 1
      skip_l2 = 1
    else:
      if max_segments < 1:
        raise ValueError, '_CURVE_DIST_MAX_SEGMENTS must be at least 1'
      skip_l1 = ((len(l1_curve) - 2) / max_segments) + 1
      skip_l2 = ((len(l2_curve) - 2) / max_segments) + 1
    
    l2_subset = range(0, len(l2_curve), skip_l2)
    l2_curve = l2_curve[l2_subset]
    for i in range(0, len(l1_curve) - 1, skip_l1):
      lamb = l1_lamb[i+1] - l1_lamb[i]
      min_d = self._distancePointToLineSegment(l2_curve[0], l2_curve[1], l1_curve[i])[0]
      for j in range(1, len(l2_curve) - 1):
        d = self._distancePointToLineSegment(l2_curve[j], l2_curve[j+1], l1_curve[i])[0] 
        if d < min_d:
          min_d = d
      weighted_dist = weighted_dist + (lamb * min_d)
    return weighted_dist / l1_lamb[-(1 + (len(l1_curve)-2)%skip_l1)]
    
  def _distancePointToLineSegment(self, a, b, p):
    '''
    Returns tuple of minimum distance to the directed line segment AB from p, and the distance along AB of the point of intersection
    '''  
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
  def setDataPoints(self, X):
    if len(X) == 0:
      raise ValueError, 'There must be at least 1 data point'
    self._X = X
    self._treeX = KDTree(X)
  def __init__(self, X, **params):
    '''
    Constructor
    '''
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
    self.setDataPoints(X)
    self._maxSegmentLength = None
  
  def calculateCoverageIndices(self, curves, tau):  
    '''
    Calculates the indices of self._X that contain points within tau of the curve defined by lpc_points.
    curves, either a single lpc curve dictionary or a list of lpc curve dictionaries
    '''
    if type(curves) == dict:
      curves = [curves]
      
    points_in_tube = set()
    
    for curve in curves:    
      indices = self._calculateNNBallIndices(curve, self._calculateBallRadius(curve, tau))
      lpc_points = curve['save_xd']
      num_lpc_pts = len(lpc_points)
            
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
 
  def getCoverageGraph(self, curves, tau_range):
    '''Return a 2*len(tau_range) array of the proportion of self._X points within tau (in tau_range, an array)
    of the curve segments, where 'curves' is a either a list of curve dictionaries as returned by LPCImpl.lpc, or an element thereof
    This should give graphs similar to the output of BOakley
    '''
    coverage = [1.0*len(self.calculateCoverageIndices(curves,tau))/len(self._X) for tau in tau_range]
    return array([tau_range,coverage])
  
  def getTubeResiduals(self, curve):
    return self._calculatePointResiduals(curve, self._params['tube_radius'])
  
  def getGlobalResiduals(self, curve):
    '''
    '''
    return self._calculatePointResiduals(curve)
  
  def _calculatePointResiduals(self, curve, tube_radius = None):
    if tube_radius is None:
      X = self._X
    else:
      within_tube_indices = self.calculateCoverageIndices(curve, tube_radius)
      X = self._X.take(list(within_tube_indices), axis = 0) 
      
    if self._maxSegmentLength is None:
      self._maxSegmentLength = self._calculateMaxSegmentLength(curve)
    lpc_points = curve['save_xd']
    num_lpc_points = len(lpc_points)
    tree_lpc_points = KDTree(lpc_points)
    residuals = empty(len(X))
    residuals_lamb = empty(len(X))
    path_length = curve['lamb']
    
    for j, p in enumerate(X): 
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
      