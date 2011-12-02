'''
Created on 22 Nov 2011

@author: droythorne
'''
from collections import defaultdict
from itertools import cycle
from latte.io_formats import LamuFileReader
from latte.io_formats.decorators import LamuRun
from latte.spatial.dok import dok
from lpcm.lpc import LPCImpl
from lpcm.lpcAnalysis import LPCCurvePruner
from lpcm.lpcDiagnostics import LPCResiduals, LPCResidualsRunner
from lpcm.lpcStartPoints import lpcMeanShift
from mpl_toolkits.mplot3d import axes3d, Axes3D
from numpy.core.numeric import array
from numpy.ma.extras import unique
from pprint import pprint
import cPickle
import matplotlib.pyplot as plt


class ToyTracks():
  '''
  Imports a toy tracks from file using  Toy tracks throw away all the energy deposition and 
  particle id information to yield what just a three dimensional set of points. 
  '''
  def __init__(self, filename):
    self._reader = LamuFileReader(filename)
  
  def unique_rows(self, data):
    unique = defaultdict(int)
    for row in data:
        unique[tuple(row)] += 1
    return array([list(e) for e in unique.keys()])
    
  def getEventHits(self, max_num_events = None):
    '''Generator of a 3 dimensional array of points from events returned by LamuFileReader hits
    '''
    if max_num_events is None:
      event_iter =  xrange(self._reader.number_of_events())
    else:
      event_iter =  xrange(min(max_num_events, self._reader.number_of_events())) 
    for index in event_iter: 
      hits = self._reader.hits(index)
      data = [[float(h.i), float(h.j), float(h.k)] for h in hits]
      unique_points = self.unique_rows(data)
      yield unique_points

class ToyTracksLamuRun():
  def __init__(self, filename, max_num_events = None):
    self._reader = LamuRun(filename)
    self._event_gen = self._reader.events()
    self.max_num_events = max_num_events
  
  def getEvent(self):
    if self.max_num_events is None:
      event_iter =  xrange(self._reader.number_of_events())
    else:
      event_iter =  xrange(min(self.max_num_events, self._reader.number_of_events()))
    for index in event_iter:     
      event = LamuEventAccessor(self._event_gen.next())
      yield event
  
class LamuEventAccessor():
   
  def __init__(self, event):
    self._event = event
  
  def getEventHits(self): 
      hits = self.getEventHitsNTuple()
      data = [[float(h[0][0]), float(h[0][1]), float(h[0][2])] for h in hits]
      data = array(data)
      return data
  
  def getEventHitsNTuple(self):
      hits = self._event['raw']['hits']
      return hits
  
  def getParticleHits(self, pdg_codes = None):
    if pdg_codes == None:
      pdg_codes = self._event['raw']['track_id'].values()
    particle_dict = {}
    all_track_ids = self._event['raw']['track_id'].keys()
    particle_list = self._event['raw']['track_id'].values()
    for particle_id in pdg_codes:
      i = -1
      particle_track_ids = []
      try:
        while 1:
              i = particle_list.index(particle_id, i+1)
              particle_track_ids.append(all_track_ids[i])
      except ValueError:
        pass
      d = dok()
      for id in particle_track_ids:
        track = self._event['raw']['truth_hits'][id]
        for h in track:
          d.add_to_voxel(h[0], h[1])
      particle_dict[particle_id] = d
      
    return particle_dict   
  def getParticlesInVoxelDict(self):
    voxel_dict = {}
    for k in self._event['raw']['truth_hits'].keys():
      pdg_code = self._event['raw']['track_id'][k] 
      for h in self._event['raw']['truth_hits'][k]:
        try:
          voxel_dict[h[0]].append(pdg_code)
        except KeyError:
          voxel_dict[h[0]] = [pdg_code]
    return voxel_dict

class ToyLPCRunner():
  
  def __init__(self, toy_tracks, target_filename_prefix, lpc_algorithm):
    self.toy_tracks = toy_tracks
    self._lpcAlgorithm = lpc_algorithm
    self.target_filename_prefix = target_filename_prefix
  def __call__(self):
    '''Pickles the lpc curves (actually a dictionary containing the event_id, the points data (perhaps scaled) and lpc_curves) of each event to a file, with filename 
    determined by the target_filename_prefix concatenated with a hash of the lpc parameters. Dumps the parameters used to generate 
    lpc curves to a file that contains a dictionary containing the lpc_parameters and the filename of the pickled object containing 
    all the event's lpc curves.
    '''
    events = self.toy_tracks.getEvent()
    lpc = self._lpcAlgorithm
    params = lpc._lpcParameters
    param_hash = hash(str(params))
    prefix = self.target_filename_prefix + '_' + str(param_hash)
    metadata_filename = prefix + '_meta.pkl'
    pk_metadata = open(metadata_filename, 'w')
    cPickle.dump({'lpc_filename': prefix + '.pkl', 'lpc_parameters': params}, pk_metadata, 0)
    pk_metadata.close()
     
    pk_lpc_data = open(prefix + '.pkl', 'w')
    
    i = 0  
    for event in events:
      lpc_curve = lpc.lpc(X=event.getEventHits())
      lpc_data = {'id': i, 'lpc_curve': lpc_curve, 'Xi': lpc.Xi, 'data_range': lpc._dataRange}
      cPickle.dump(lpc_data, pk_lpc_data, -1)
      i = i + 1
      
    pk_lpc_data.close()
    return metadata_filename

def toyLPCPlotter(Xi, lpc_curve):
  '''
  Draws a lovely 3d picture of the clusters, start points and resutling lpc curves, colour coded by cluster that seeded them
  '''
  plt.close('all')
  fig = plt.figure(0)
  fig_angle = plt.figure(1)
  fig_rho = plt.figure(2)
  fig.clear()
  ax = Axes3D(fig)
  ax.set_title('toyTrack')
  
  colors = cycle('bgrcmyk')
  plt.figure(0)
  ax.scatter(Xi[:, 0], Xi[:, 1], Xi[:, 2], c = 'black', alpha = 0.1)
    
  for k, col in zip(range(len(lpc_curve)), colors):  
    start_point = lpc_curve[k]['start_point']
    ax.scatter([start_point[0]],[start_point[1]], [start_point[2]], c = col, marker = '^', s = [10])
    curve = lpc_curve[k]['save_xd']
    ax.plot(curve[:,0],curve[:,1],curve[:,2], c = col, linewidth = 3, alpha = 0.5)
    plt.figure(1)
    plt.plot(lpc_curve[k]['lamb'], lpc_curve[k]['cos_neu_neu'], color = col)
    plt.figure(2)
    plt.plot(lpc_curve[k]['lamb'], lpc_curve[k]['rho'], color = col)
  fig.canvas.draw()
  plt.show()      
  
def LPCEfficiencyCalculator(curves, data_range, particle_hits, tau):
  #first rescale the curves then calculate the proton and muon efficiency
  muon_hits = array([[h[0][0], h[0][1], h[0][2]] for h in particle_hits[13]]) / data_range
  proton_hits = array([[h[0][0], h[0][1], h[0][2]] for h in particle_hits[2212]]) / data_range
  residuals_calc = LPCResiduals(muon_hits, tube_radius = 0.12)
  residuals_runner = LPCResidualsRunner(curves, residuals_calc)
  residuals_runner.setTauRange([tau])
  muon_residuals = residuals_runner.calculateResiduals(True, False, False)
  muon_coverage = max([len(muon_residuals['curve_residuals'][i]['coverage_indices'][tau]) for i in range(len(curves))])
  muon_efficiency = float(muon_coverage) / len(muon_hits)
  
  residuals_calc.setDataPoints(proton_hits)
  proton_residuals = residuals_runner.calculateResiduals(True, False, False)
  proton_coverage = max([len(proton_residuals['curve_residuals'][i]['coverage_indices'][tau]) for i in range(len(curves))])
  proton_efficiency = float(proton_coverage) / len(proton_hits)
  print 'residuals'

def LPCPurityCalculator(curves, data_range, voxel_to_pdg_dictionary, tau):
  hit_tuples = voxel_to_pdg_dictionary.keys()
  hits = array([[h[0], h[1], h[2]] for h in hit_tuples]) / data_range
  residuals_calc = LPCResiduals(hits, tube_radius = 0.12)
  residuals_runner = LPCResidualsRunner(curves, residuals_calc)
  residuals_runner.setTauRange([tau])
  residuals = residuals_runner.calculateResiduals(True, False, False)
  pdg_code_frequencies = []
  for i in range(len(curves)):
    d = defaultdict(int)
    hit_labels = [voxel_to_pdg_dictionary[hit_tuples[i]] for i in residuals['curve_residuals'][i]['coverage_indices'][tau]]
    flattened_hit_labels = [pdg_code for pdg_code_list in hit_labels for pdg_code in pdg_code_list]
    for pdg_code in flattened_hit_labels:
      d[pdg_code] += 1
    pdg_code_frequencies.append(d)
  return pdg_code_frequencies
def toyLPCCurveDump(rootfile, outfile):
  lpc = LPCImpl(start_points_generator =  lpcMeanShift(ms_h = 0.27), h = 0.02, t0 = 0.03, pen = 3, it = 100, mult = 5, scaled = True, cross = False, convergence_at = 0.0001)
  rootfile = "/home/droythorne/Downloads/muon-proton.root"
  outfile = '/tmp/muon-proton'
  #read root events, calculate lpc_curves and pickle curve and parameter data
  toy = ToyTracksLamuRun(rootfile, max_num_events=100)
  runner = ToyLPCRunner(toy, outfile, lpc)
  metadata_filename = runner() 
  return metadata_filename  

class toyLPCCurveLoad():
  def __init__(self, metadata_filename):
    self.metadata_filename = metadata_filename
    f = open(self.metadata_filename, 'rb')
    batch_parameters = cPickle.load(f)
    self.curves_filename = batch_parameters['lpc_filename']
    pprint(batch_parameters['lpc_parameters'])
    f.close()
    self._f = open(self.curves_filename, 'rb')
  def __del__(self):
    self._f.close()
  def getEvent(self):   
    evt_curves = cPickle.load(self._f)
    return evt_curves  

if __name__ == '__main__':
  '''
  Example of how to read in a root file containing event with hits, generate lpc_curves for each event, serialise these using cPickle, then 
  read the events back in to plot
  '''
  rootfile = "/home/droythorne/Downloads/muon-proton.root"
  outfile = '/tmp/muon-proton'
  #metadata_filename = toyLPCCurveDump(rootfile, outfile)
  #read in parameters from an event batch, print them, then plot the first event from that batch
  metadata_filename = '../resources/muon-proton_6659714327790660657_meta.pkl'
  lpc_curve_load = toyLPCCurveLoad(metadata_filename)
  
  toy_truth = ToyTracksLamuRun(rootfile)
  truth_gen = toy_truth.getEvent()
  #toyLPCPlotter(evt_curves['Xi'], evt_curves['lpc_curve'])
  out_data = []
  while 1:
    try:
      evt = lpc_curve_load.getEvent()
      truth_evt = truth_gen.next()
      #pprint(truth_evt.getParticlesInVoxelDict())
      #now calcualte the residuals
      
      residuals_calc = LPCResiduals(evt['Xi'], tube_radius = 0.12)
      residuals_runner = LPCResidualsRunner(evt['lpc_curve'], residuals_calc)
      residuals_runner.setTauRange([0.07])
      
      pruner = LPCCurvePruner(residuals_runner)
      remaining_curves = pruner.pruneCurves()
      toyLPCPlotter(evt['Xi'], remaining_curves)
  
      tau = 0.015
      #muon_proton_hits = truth_evt.getParticleHits([13, 2212])
      #eff = LPCEfficiencyCalculator(remaining_curves, evt['data_range'], muon_proton_hits, tau)
      voxel_to_pdg_dictionary = truth_evt.getParticlesInVoxelDict()
      pur = LPCPurityCalculator(remaining_curves, evt['data_range'], voxel_to_pdg_dictionary, tau) 
      
      out_data.append({'voxel_dict': voxel_to_pdg_dictionary, 'pur': pur})
      print 'breakpoint'
    except EOFError:
      break
    
  outfile = open('/home/droythorne/purity_data_1.pkl', 'w')
  cPickle.dump(out_data, outfile, -1)