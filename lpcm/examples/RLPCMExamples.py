'''
Created on 21 Nov 2011
@author: droythorne
RLPCExamples - recreates the examples contain in the original LPCM R code documentation
'''
from lpcm.lpc import LPCImpl
from lpcm.lpcStartPoints import lpcMeanShift
from mpl_toolkits.mplot3d.axes3d import Axes3D
from numpy.core.numeric import array
import matplotlib.pyplot as plt
def readData(filename):
  '''
  Reads text file, filename,  containing an arbitrary number of columns of data that may be coerced to floats. Returns a list of lists containing the data from each row.
  '''
  f = open(filename)
  x = []
  for l in f.readlines():
    y = [float(v) for v in l.split()]
    x.append( y )
  f.close()
  return x

def calSpeedFlow():
  calspeedflow = array(readData('../resources/calspeedflow.dat'))
  print calspeedflow
  lpc = LPCImpl(start_points_generator = lpcMeanShift(ms_h = 0.1), mult = 1)
  lpc.lpc(X = calspeedflow)
  curve = lpc.getCurve(unscale = True)
  fig = plt.figure()
  plt.scatter(calspeedflow[:,0], calspeedflow[:,1], alpha = 0.3)
  save_xd = curve[0]['save_xd']
  plt.plot(save_xd[:,0], save_xd[:,1])
  
  
def gaia():
  gaia = array(readData('../resources/gaia.dat'))
  lpc = LPCImpl(start_points_generator = lpcMeanShift(), mult = 1, scaled = False)
  curve = lpc.lpc(X = gaia)
  fig = plt.figure()
  ax = Axes3D(fig)
  ax.scatter(gaia[:,0], gaia[:,1], gaia[:,2], alpha = 0.3)
  save_xd = curve[0]['save_xd']
  ax.plot(save_xd[:,0], save_xd[:,1], save_xd[:,2])
  
  
if __name__ == '__main__':
  calSpeedFlow()
  gaia()
  plt.show()
  