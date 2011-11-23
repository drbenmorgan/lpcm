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
import cPickle
import matplotlib.pyplot as plt

class ToyTracks():
  '''
  Imports a toy tracks from file using latte filereader and applies lpc algorithm (FILL IN DETAILS!)  
  '''
  def __init__(self, filename, lpc_algorithm):
    self._reader = LamuFileReader(filename)
    self._lpcAlgorithm = lpc_algorithm
  
  def unique_rows(self, data):
    unique = defaultdict(int)
    for row in data:
        unique[tuple(row)] += 1
    return array([list(e) for e in unique.keys()])
    
  def getEventHits(self):
    for index in xrange(min(5, self._reader.number_of_events())):   
      hits = self._reader.hits(index)
      data = [[float(h.i), float(h.j), float(h.k)] for h in hits]
      unique_points = self.unique_rows(data)
      yield unique_points

def toyLPCPlotter(Xi, lpc_curve):
  '''
  Draws a lovely 3d picture of the clusters, start points and resutling lpc curves, colour coded by cluster that seeded them
  '''
  fig = plt.figure()
  ax = Axes3D(fig)
  ax.set_title('toyTrack')
  labels = lpc._startPointsGenerator._meanShift.labels_
  labels_unique = unique(labels)
  n_clusters = len(labels_unique)
  colors = cycle('bgrcmyk')
  for k, col in zip(range(n_clusters), colors):
    cluster_members = labels == k
    ax.scatter(Xi[cluster_members, 0], Xi[cluster_members, 1], Xi[cluster_members, 2], c = col, alpha = 0.1)
    start_point = lpc_curve[k]['start_point']
    ax.scatter([start_point[0]],[start_point[1]], [start_point[2]], c = col, marker = '^', s = [10])
    curve = lpc_curve[k]['save_xd']
    ax.plot(curve[:,0],curve[:,1],curve[:,2], c = col, linewidth = 3, alpha = 0.5)
  plt.show()                  

def toyLPCRunner(toy, target_filename):
  event_gen = toy.getEventHits()
  lpc = toy._lpcAlgorithm
  pk_file = open(target_filename, 'w')  
  for event in event_gen:
    lpc_curve = lpc.lpc(X=event)
    lpc_data = {'lpc_curve': lpc_curve, 'Xi': lpc.Xi}
    cPickle.dump(lpc_data, pk_file, -1)
  #toyLPCPlotter(lpc.Xi, lpc_curve)
  
if __name__ == '__main__':
  lpc = LPCImpl(start_points_generator = lpcMeanShift(ms_h = 0.25), h = 0.03, t0 = 0.05, pen = 3, it = 500, scaled = True, cross = False, convergence_at = 0.0001)
  rootfile = "/home/droythorne/Downloads/muon-proton.root"
  outfile = '/tmp/muon.pkl'
  toy = ToyTracks(rootfile, lpc)
  toyLPCRunner(toy, outfile )
  #lpc_curve = lpc.lpc(X=line)  

  