'''
Created on 22 Nov 2011

@author: droythorne
'''
from collections import defaultdict
from itertools import cycle
from lpcm.lpcAnalysis import lpcCurvePruner, lpcAnalysisPickleReader
from lpcm.lpcDiagnostics import LPCResiduals, LPCResidualsRunner
from lpcm.lpcProcessing import LamuRead, lpcProcessor
from mpl_toolkits.mplot3d import axes3d, Axes3D
from numpy.core.numeric import array
import cPickle
import matplotlib.pyplot as plt
import sys

class LamuEventAccessor():
  '''Moved to lpcProcessing.LamuEventDecorator
  ''' 
  pass
class ToyLPCRunner():
  '''Moved to lpcProcessing.LPCProcessor
  '''
  pass

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
  if data_range is None:
    data_range = 1.0
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
  if data_range is None:
    data_range = 1.0
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
  '''Moved to lpcProcessing.LpcPLPCBaseWriter and derived classes
  '''
  pass

class toyLPCCurveLoad():
  '''DEPRACATED - use lpcAnalysisPickleReader
  Currently only works on pickled data
  TODO - implement a shelved data version  '''
  def __init__(self, metadata_filename):
    self.metadata_filename = metadata_filename
    f = open(self.metadata_filename, 'rb')
    self.batch_parameters = cPickle.load(f)
    self.curves_filename = self.batch_parameters['lpc_filename']
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
  if len(sys.argv) != 2:
    raise ValueError, 'Must supply the name of a configuration file'
  proc = lpcProcessor(sys.argv[1])
  proc.runProcessor()
  metadata_filename = proc.getMetadataFilename()
  lpc_loader = lpcAnalysisPickleReader(metadata_filename)
  #toyLPCPlotter(evt_curves['Xi'], evt_curves['lpc_curve'])
  out_data = []
  while 1:
    try:
      evt = lpc_loader.getEvent()
      #pprint(truth_evt.getParticlesInVoxelDict())
      #now calcualte the residuals
      residuals_calc = LPCResiduals(evt[0]['Xi'], tube_radius = 3.0)
      residuals_runner = LPCResidualsRunner(evt[0]['lpc_curve'], residuals_calc)
      residuals_runner.setTauRange([2.0])
      
      pruner = lpcCurvePruner(residuals_runner, closeness_threshold = 5.0, path_length_threshold = 10.0)
      remaining_curves = pruner.pruneCurves()
      tau = 2.0
      #muon_proton_hits = truth_evt.getParticleHits([13, 2212])
      #eff = LPCEfficiencyCalculator(remaining_curves, evt['data_range'], muon_proton_hits, tau)
      voxel_to_pdg_dictionary = evt[1].getParticlesInVoxelDict()
      pur = LPCPurityCalculator(remaining_curves, evt[0]['data_range'], voxel_to_pdg_dictionary, tau) 
      
      out_data.append({'voxel_dict': voxel_to_pdg_dictionary, 'pur': pur})
      print 'breakpoint'
    except EOFError:
      break
    
  outfile = open('/tmp/purity_data.pkl', 'w')
  cPickle.dump(out_data, outfile, -1)