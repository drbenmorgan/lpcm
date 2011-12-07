'''
Created on 25 Nov 2011

@author: droythorne
'''
from collections import defaultdict
from lpcm.lpcDiagnostics import LPCResiduals, LPCResidualsRunner
from lpcm.lpcParser import lpcAnalysisParser
from lpcm.lpcProcessing import LamuRead
from numpy.core.numeric import array
from scitools.PrmDictBase import PrmDictBase
import cPickle
import shelve

class lpcAnalysisBaseReader(object):
  def __init__(self, metadata_filename, max_events):
    '''NB max_events actually has no influence on the operation of the class for the moment
    '''
    self.metadata_filename = metadata_filename
    self._max_events = max_events
    f = open(self.metadata_filename, 'rb')
    self.batch_parameters = cPickle.load(f)
    self.curves_filename = self.batch_parameters['lpc_filename']
    f.close()
    
    self._truth = LamuRead(self.batch_parameters['reader_parameters']['filename'], self.batch_parameters['reader_parameters']['max_events'])
  def __del__(self):
    self._f.close()

class lpcAnalysisPickleReader(lpcAnalysisBaseReader):
  def __init__(self, metadata_filename, max_events = None):
    lpcAnalysisBaseReader.__init__(self, metadata_filename, max_events)
    self._truth_generator = self._truth.getEventGenerator()
    self._f = open(self.curves_filename, 'rb')
  
  def getNextEvent(self):
    '''Returns a tuple of (evt_curves, evt_truth), where evt_curves returns the next event from the pickled
    output of lpcProcessing, and evt_truth is a LamuEventDecorator object. The LamuEventDecorator will 
    contain the relevant truth data for the loaded pickled event, on the assumption that this object references
    the unaltered root files, metadata and output from lpcProcessing. 
    '''  
    evt_curves = cPickle.load(self._f)
    evt_truth = self._truth_generator.next()
    return evt_curves, evt_truth

class lpcAnalysisShelveReader(lpcAnalysisBaseReader):
  def __init__(self, metadata_filename, max_events = None):
    lpcAnalysisBaseReader.__init__(self, metadata_filename, max_events)
    self._truth_generator = self._truth.getEventGenerator()
    self._evt_generator = self._eventGenerator()
    self._f = shelve.open(self.curves_filename)
    
  def _eventGenerator(self):
    '''Relies upon data having been shelved using LPCShelver (i.e. keys are indexed with integer strings from '0' to '<<num_events - 1>>'
    whereby data read in sequence using self._truth_generator and self._evt_generator will correspond to the same event
    '''  
    i = -1
    while 1:
      i += 1
      yield self._f[str(i)]
  
  def getNextEvent(self):
    '''Gets tuples in sequence with zeroth element equal to a shelved lpcProcessing event and first element a LamuEventDecorator object
    (mimics behavior of lpcAnalysisPickleReader) Once the number of calls exceed the number of shelved events, raises an EOF exception
    (as pickled events would) so that lpcAnalyser can terminate processing loop
    '''
    try:
      evt_curves = self._evt_generator.next()
    except KeyError:
      raise EOFError
    evt_truth = self._truth_generator.next()
    return evt_curves, evt_truth
  
  def getSingleEvent(self, evt_id):
    '''Gets a specific event number'''
    evt_truth = self._truth.getEvent(evt_id)
    evt_curves = self._f[str(evt_id)]
    return evt_curves, evt_truth

class lpcCurvePruner(PrmDictBase):
    '''
    Helper functions for using lpc curves and their residuals to prune similar curves from lpc output. Answers the question, 'When
    are two lpc curves the same?'
    '''
    def __init__(self, residuals_runner, **params):
      '''
      Constructor
      '''
      super(lpcCurvePruner, self).__init__()
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
        self._residuals = self._residualsRunner.getResiduals()
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

class LPCPurityCalculator(object):
  '''Calcualtes purity by event for each curve and for each value in tau_range. SHould be initialised with LPCDiagnostics.LPCResidualsRunner
  instance'''
  def __init__(self, residuals_runner):
    self._residuals_runner = residuals_runner
    
  def calculatePurity(self, curves, data_range, voxel_to_pdg_dictionary):
    '''NB - self._residuals_runner should have had calculateResiduals method called with calc_residuals = True beofre calling this method
    '''
    hit_tuples = voxel_to_pdg_dictionary.keys()
    if data_range is None:
      data_range = 1.0
    #rescales the truth data if necessary
    hits = array([[h[0], h[1], h[2]] for h in hit_tuples]) / data_range
    self._residuals_runner.setDataPoints(hits)
    self._residuals_runner.setLpcCurves(curves)
    self._residuals_runner.calculateResiduals(True, False, False)
    residuals = self._residuals_runner.getResiduals()
    tau_range = self._residuals_runner.getTauRange()
    purity = {}
    for tau in tau_range:
      pdg_code_frequencies = []
      for i in range(len(curves)):
        d = defaultdict(int)
        hit_labels = [voxel_to_pdg_dictionary[hit_tuples[i]] for i in residuals['curve_residuals'][i]['coverage_indices'][tau]]
        flattened_hit_labels = [pdg_code for pdg_code_list in hit_labels for pdg_code in pdg_code_list]
        for pdg_code in flattened_hit_labels:
          d[pdg_code] += 1
        pdg_code_frequencies.append(d)
      purity[tau] = pdg_code_frequencies
    return purity

class lpcAnalyser(object):
  def __init__(self, filename):
    '''
    Constructor
    '''
    self._parser = lpcAnalysisParser(filename)
    self._initReader()
    self._initResiduals()
    self._initPruner()
    self._setOutputFilename()
    self._initPurity()
  def _initReader(self):
    run_parameters = self._parser.getReadParameters()
    if run_parameters['type'] == 'lpcAnalysisPickleReader':
      self._reader = lpcAnalysisPickleReader(**run_parameters['params'])
    else:
      raise ValueError, 'Specified type of reader is not recognised'
  def _initResiduals(self):
    residual_parameters = self._parser.getResidualsParameters()
    if residual_parameters['type'] == 'LPCResiduals':
      tau_range = residual_parameters['params'].pop('tau_range')
      self._residuals = LPCResiduals(**residual_parameters['params'])
      self._residuals_runner = LPCResidualsRunner(self._residuals, tau_range)
    else:
      raise ValueError, 'Specified type of residuals calculator is not recognised'
  def _initPruner(self):
    pruner_parameters = self._parser.getPrunerParameters()
    if pruner_parameters['type'] == 'lpcCurvePruner':
      self._pruner = lpcCurvePruner(self._residuals_runner, **pruner_parameters['params'])
    else:
      raise ValueError, 'Specified type of residuals calculator is not recognised'
  def _setOutputFilename(self):
    misc = self._parser.getMiscParameters()
    self._output_filename = misc['params']['output_filename']
  
  def _initPurity(self):
    self._purity = LPCPurityCalculator(self._residuals_runner)
  
  def runAnalyser(self):
    out_data = []
    while 1:
      try:
        evt = self._reader.getNextEvent()
        self._residuals.setDataPoints(evt[0]['Xi'])
        self._residuals_runner.setLpcCurves(evt[0]['lpc_curve'])
        self._residuals_runner.calculateResiduals(calc_residuals = False, calc_containment_matrix = False)
        remaining_curves = self._pruner.pruneCurves()
        #muon_proton_hits = truth_evt.getParticleHits([13, 2212])
        #eff = LPCEfficiencyCalculator(remaining_curves, evt['data_range'], muon_proton_hits, tau) TODO - move from examples/toytracks 
        voxel_to_pdg_dictionary = evt[1].getParticlesInVoxelDict()
        pur = self._purity.calculatePurity(remaining_curves, evt[0]['data_range'], voxel_to_pdg_dictionary) 
        out_data.append({'voxel_dict': voxel_to_pdg_dictionary, 'pur': pur})
      except EOFError:
        break
      
    outfile = open(self._output_filename, 'w')
    cPickle.dump(out_data, outfile, -1)

if __name__ == "__main__":
  analyser = lpcAnalyser('../../resources/test_analysis.xml')
  analyser.runAnalyser()
  print 'Done!'
  
  '''
  TODO - stick this into a unit test
  Just a quick test of calculateAbsoluteScaleDistanceMatrixCurves for now. All the gumpf beforehand is
  redundant to the test, just creating a chain of objects to instantiate the lpcCurvePruner instance
  
  t =  arange(-1,1,0.1)
  line = array(zip(t,t,t))
  lpc = LPCImpl()
  lpc_curve = lpc.lpc(X=line)
  residuals_calc = LPCResiduals(tube_radius = 0.15)
  residuals_calc.setDataPoints(line)
  residuals_runner = LPCResidualsRunner(residuals_calc)
  residuals_runner.setLpcCurves(lpc_curve.getCurve())
  analysis = lpcCurvePruner(residuals_runner)   
  a = array([[ 0.,          0.00058024,  0.00078112,  0.22710005,  0.22702893],
             [ 0.00063906,  0.,          0.00029174,  0.22873423,  0.2286578 ],
             [ 0.00075869,  0.00030338,  0.,          0.23164868,  0.23158872],
             [ 0.1799655,   0.18003384,  0.1800275,  0.,          0.00043582],
             [ 0.18029958,  0.18030112,  0.18029657,  0.00034616,  0.        ]])
  analysis._residuals = {'distance_matrix': a}
  rem_curves = analysis._calculateAbsoluteScaleDistanceMatrixCurves()      
  '''
        