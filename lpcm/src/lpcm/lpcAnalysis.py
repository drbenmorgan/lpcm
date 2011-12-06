'''
Created on 25 Nov 2011

@author: droythorne
'''
from lpcm.lpc import LPCImpl
from lpcm.lpcDiagnostics import LPCResiduals, LPCResidualsRunner
from lpcm.lpcProcessing import LamuRead
from numpy.core.numeric import array
from numpy.ma.core import arange
from scitools.PrmDictBase import PrmDictBase
import cPickle

class lpcAnalysisBaseReader(object):
  def __init__(self, metadata_filename):
    self.metadata_filename = metadata_filename
    f = open(self.metadata_filename, 'rb')
    self.batch_parameters = cPickle.load(f)
    self.curves_filename = self.batch_parameters['lpc_filename']
    f.close()
    self._f = open(self.curves_filename, 'rb')
    self._truth = LamuRead(self.batch_parameters['reader_parameters']['filename'], self.batch_parameters['reader_parameters']['max_events'])
  def __del__(self):
    self._f.close()

class lpcAnalysisPickleReader(lpcAnalysisBaseReader):
  def __init__(self, metadata_filename):
    lpcAnalysisBaseReader.__init__(self, metadata_filename)
    self._truth_generator = self._truth.getEventGenerator()
  
  def getEvent(self):
    '''Returns a tuple of (evt_curves, evt_truth), where evt_curves returns the next event from the pickled
    output of lpcProcessing, and evt_truth is a LamuEventDecorator object. The LamuEventDecorator will 
    contain the relevant truth data for the loaded pickled event, on the assumption that this object references
    the unaltered root files, metadata and output from lpcProcessing. 
    '''  
    evt_curves = cPickle.load(self._f)
    evt_truth = self._truth_generator.next()
    return evt_curves, evt_truth

class lpcAnalysisShelveReader(lpcAnalysisBaseReader):
  pass

class LPCCurvePruner(PrmDictBase):
    '''
    Helper functions for using lpc curves and their residuals to prune similar curves from lpc output. Answers the question, 'When
    are two lpc curves the same?'
    '''
    def __init__(self, residuals_runner, **params):
      '''
      Constructor
      '''
      super(LPCCurvePruner, self).__init__()
      self._params = {  'closeness_threshold': 0.1, #how close before the curve is thrown away 
                        'path_length_threshold': 0.05 #reject all curves with path length less than this
                     }
      self._prm_list = [self._params] 
      self.user_prm = None #extension of parameter set disallowed
      self._type_check.update({ 'closeness_threshold': lambda x: isinstance(x, float) and x > 0,
                                'path_length_threshold': lambda x: isinstance(x, float) and x > 0  
                              })
      self.set(**params)
      self._residualsRunner = residuals_runner
      self._residuals = None
      self._retained_curves = None
      
    def _calculateAbsoluteScaleDistanceMatrixCurves(self):
      if self._residuals is None:
        self._residuals = self._residualsRunner.calculateResiduals(calc_residuals = False, calc_containment_matrix = False)
      dm = self._residuals['distance_matrix']
      dm_list = []
      for i in range(dm.shape[0]):
        for j in range(dm.shape[1]):
          if i != j:
            dm_list.append((dm[i,j],(i,j)))
      dm_list.sort()      
      tau = self._params['closeness_threshold']
      removed_curves = []
      for e in dm_list:
        if e[0] < tau and (e[1][0] not in removed_curves) and (e[1][1] not in removed_curves):
          removed_curves.append(e[1][0])
      return removed_curves
    def getRetainedCurveIndices(self):
      return self._retained_curves
    def pruneCurves(self):
      curves = self._residualsRunner._lpcCurves
      short_curves = []
      for i, curve in enumerate(curves):
        if curve['lamb'][-1] < self._params['path_length_threshold']:
          short_curves.append(i)
      self._residualsRunner._lpcCurves = [curves[i] for i in range(len(curves)) if i not in short_curves]    
      
      rem_curves = self._calculateAbsoluteScaleDistanceMatrixCurves()
      num_curves = len(self._residualsRunner._lpcCurves)
      remaining_curve_indices = list(set(range(0, num_curves)) - set(rem_curves)) #TODO, betterify this monstrosity
      self._retained_curves = remaining_curve_indices
      remaining_curves = []
      for i in remaining_curve_indices:
        remaining_curves.append(self._residualsRunner._lpcCurves[i]) #pretty inefficient to copy this
      return remaining_curves

if __name__ == "__main__":
  '''Just a quick test of calculateAbsoluteScaleDistanceMatrixCurves for now. All the gumpf beforehand is
  redundant to the test, just creating a chain of objects to instantiate the LPCCurvePruner instance
  '''
  t =  arange(-1,1,0.1)
  line = array(zip(t,t,t))
  lpc = LPCImpl()
  lpc_curve = lpc.lpc(X=line)
  residuals_calc = LPCResiduals(line, tube_radius = 0.15)
  residuals_runner = LPCResidualsRunner(lpc.getCurve(), residuals_calc)
  analysis = LPCCurvePruner(residuals_runner)   
  a = array([[ 0.,          0.00058024,  0.00078112,  0.22710005,  0.22702893],
             [ 0.00063906,  0.,          0.00029174,  0.22873423,  0.2286578 ],
             [ 0.00075869,  0.00030338,  0.,          0.23164868,  0.23158872],
             [ 0.1799655,   0.18003384,  0.1800275,  0.,          0.00043582],
             [ 0.18029958,  0.18030112,  0.18029657,  0.00034616,  0.        ]])
  analysis._residuals = {'distance_matrix': a}
  rem_curves = analysis._calculateAbsoluteScaleDistanceMatrixCurves()      
        