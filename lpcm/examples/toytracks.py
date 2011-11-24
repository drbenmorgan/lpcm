'''
Created on 22 Nov 2011

@author: droythorne
'''
from collections import defaultdict
from itertools import cycle
from latte.io_formats import LamuFileReader
from lpcm.lpc import LPCImpl
from lpcm.lpcStartPoints import lpcMeanShift
from mpl_toolkits.mplot3d import axes3d, Axes3D
from numpy.core.numeric import array
from numpy.ma.extras import unique
from pprint import pprint
import cPickle
import matplotlib.pyplot as plt


class ToyTracks():
  '''
  Imports a toy tracks from file using latte's LamuFileReader and applies lpc algorithm. Toy tracks throw away all the energy deposition and 
  particle id information to yield what just a three dimensional set of points. 
  '''
  def __init__(self, filename, lpc_algorithm):
    self._reader = LamuFileReader(filename)
    self._lpcAlgorithm = lpc_algorithm
  
  def unique_rows(self, data):
    unique = defaultdict(int)
    for row in data:
        unique[tuple(row)] += 1
    return array([list(e) for e in unique.keys()])
    
  def getEventHits(self, max_num_events = None):
    '''Generator of a 3 dimensional array of points from events returned by LamuFileReader hits
    '''
    if max_num_events is None:
      event_iter =  xrange(5, self._reader.number_of_events())
    else:
      event_iter =  xrange(min(max_num_events, self._reader.number_of_events())) 
    for index in event_iter:   
      hits = self._reader.hits(index)
      data = [[float(h.i), float(h.j), float(h.k)] for h in hits]
      unique_points = self.unique_rows(data)
      yield unique_points



class toyLPCRunner():
  
  def __init__(self, toy_tracks, target_filename_prefix):
    self.toy_tracks = toy_tracks
    self.target_filename_prefix = target_filename_prefix
  def __call__(self, max_num_events = None):
    '''Pickles the lpc curves (actually a dictionary containing the event_id, the points data (perhaps scaled) and lpc_curves) of each event to a file, with filename 
    determined by the target_filename_prefix concatenated with a hash of the lpc parameters. Dumps the parameters used to generate 
    lpc curves to a file that contains a dictionary containing the lpc_parameters and the filename of the pickled object containing 
    all the event's lpc curves.
    '''
    event_gen = self.toy_tracks.getEventHits(max_num_events)
    lpc = self.toy_tracks._lpcAlgorithm
    params = lpc._lpcParameters
    param_hash = hash(str(params))
    prefix = self.target_filename_prefix + '_' + str(param_hash)
    metadata_filename = prefix + '_meta.pkl'
    pk_metadata = open(metadata_filename, 'w')
    cPickle.dump({'lpc_filename': prefix + '.pkl', 'lpc_parameters': params}, pk_metadata, 0)
    pk_metadata.close()
     
    pk_lpc_data = open(prefix + '.pkl', 'w')
    
    i = 0  
    for event in event_gen:
      lpc_curve = lpc.lpc(X=event)
      lpc_data = {'id': i, 'lpc_curve': lpc_curve, 'Xi': lpc.Xi}
      cPickle.dump(lpc_data, pk_lpc_data, -1)
      i = i + 1
      
    pk_lpc_data.close()
    return metadata_filename

def toyLPCPlotter(Xi, lpc_curve):
  '''
  Draws a lovely 3d picture of the clusters, start points and resutling lpc curves, colour coded by cluster that seeded them
  '''
  fig = plt.figure(0)
  fig_angle = plt.figure(1)
  fig_rho = plt.figure(2)
  fig.clear()
  ax = Axes3D(fig)
  ax.set_title('toyTrack')
  
  colors = cycle('bgrcmyk')
  for k, col in zip(range(len(lpc_curve)), colors):
    plt.figure(0)
    ax.scatter(Xi[:, 0], Xi[:, 1], Xi[:, 2], c = col, alpha = 0.1)
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
           
if __name__ == '__main__':
  '''
  Example of how to read in a root file containing event with hits, generate lpc_curves for each event, serialise these using cPickle, then 
  read the events back in to plot
  '''
  lpc = LPCImpl(start_points_generator = lpcMeanShift(ms_h = 0.27), h = 0.02, t0 = 0.03, pen = 3, it = 500, scaled = True, cross = False, convergence_at = 0.0001)
  rootfile = "/home/droythorne/Downloads/muon-proton.root"
  outfile = '/tmp/muon-proton'
  
  #read root events, calculate lpc_curves and pickle curve and parameter data
  toy = ToyTracks(rootfile, lpc)
  runner = toyLPCRunner(toy, outfile)
  num_events = 2
  metadata_filename = runner(num_events)
  
  #read in parameters from an event batch, print them, then plot the first event from that batch
  
  f = open(metadata_filename, 'rb')
  batch_parameters = cPickle.load(f)
  curves_filename = batch_parameters['lpc_filename']
  pprint(batch_parameters['lpc_parameters'])
  f.close()
  f = open(curves_filename, 'rb')
  evt_curves = cPickle.load(f)
  toyLPCPlotter(evt_curves['Xi'], evt_curves['lpc_curve'])

  