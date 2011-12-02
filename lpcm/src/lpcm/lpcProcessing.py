'''
Created on 1 Dec 2011

@author: droythorne
Classes for reading ROOT files, running the lpc algorithm and serialising the initial lpc output
'''
from elementtree.ElementTree import ElementTree
from latte.io_formats.decorators import LamuRun
from latte.spatial.dok import dok
from lpcm.lpc import LPCImpl
from lpcm.lpcStartPoints import lpcMeanShift
from numpy.ma.core import array
import cPickle
import os
import shelve

class LPCParameterParser(object):
  '''
  Returns dictionary with keys 'type' and 'params'
  '''
  def __init__(self, filename):
    '''
    Parameters
    ----------
    filename : path to xml config file (TODO - SCHEMA definition)
    '''
    self._filename = filename
    self._config_tree = ElementTree(file = filename)
  
  def _generateParamDictionary(self, tag):
    '''Generates a dictionary containing 'type', a string defining the 'type' attribute of element 'tag'
    and 'params', a dictionary of parameters to be unpacked as arguments to constructors for instances of 'type'
    '''
    type_node = self._config_tree.getiterator(tag)
    if len(type_node) == 1:
      type = type_node[0].get('type')
      params = {}
      for par in type_node[0]:  
        items = dict(par.items())
        s = 'v=' + items['type'] + '("' + items['value'] + '")' 
        exec(s) #TODO - remove exec, there are clearly better ways to do this!
        params[items['name']] = v
      return {'type': type, 'params': params}
    else:
      msg = 'The required lpc configuration element tag, ' + tag + ' is missing from ' + self._filename
      raise ValueError, msg
  
  def getRunParameters(self):
    '''Gets the filename of event file, type that reads it in and, if present, max number of 
    events to process (set as None if absent)
    'max_events', 'type', 'filename'
    '''
    d = self._generateParamDictionary('Run')
    return d
  def getStartPointsParameters(self):
    '''Return a dictionary containing the type name and constructor parameters for the start points generator
    'type'
    '''
    d = self._generateParamDictionary('StartPoints')
    return d
  def getLpcParameters(self):
    '''Parameters for the LPCImpl constructor
    '''
    d = self._generateParamDictionary('Lpc')
    return d
  def getSerialisationParameters(self):
    '''Parameters that govern the output directory, filename prefix, content?...'''
    d = self._generateParamDictionary('Serialization')
    return d

class LamuEventDecorator(object):
  '''
  '''
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
  
class LamuRead(object):
  '''
  '''
  def __init__(self, filename, max_events = None):
    self.filename = filename
    self._reader = LamuRun(filename)
    self._event_gen = self._reader.events()
    self.max_events = max_events
  
  def getEventGenerator(self):
    if self.max_events is None:
      event_iter =  xrange(self._reader.number_of_events())
    else:
      event_iter =  xrange(min(self.max_events, self._reader.number_of_events()))
    for index in event_iter:     
      event = LamuEventDecorator(self._event_gen.next())
      yield event    
  
  def getParameters(self):
    return {'max_events': self.max_events, 'filename': self.filename}


class LPCBaseWriter(object):
  '''Derived classes should implement a 'writeEvent' method that takes an event_id as first argument and 
  serializable event data as second
  '''
  def __init__(self, reader, lpc_algorithm, target_directory, target_filename_prefix):
    lpc_parameters = lpc_algorithm._lpcParameters
    reader_parameters = reader.getParameters()
    param_hash = hash(str((lpc_parameters, reader_parameters)))
    prefix = os.path.join(target_directory, target_filename_prefix + '_' + str(param_hash))
    #dump metadata to file
    self._metadata_filename = self.metadataPickler(lpc_parameters, reader_parameters, prefix, self.__class__.SUFFIX)
    #open file for wrting event data
    self.open(prefix)
  
  def open(self, prefix):  
    self._lpc_data = open(prefix + '.pkl', 'w')
  
  def close(self):  
    self._lpc_data.close()
  
  def metadataPickler(self, lpc_parameters, reader_parameters, prefix, suffix):
    #dump metadata to file
    _metadata_filename = prefix + '_meta.pkl'
    pk_metadata = open(_metadata_filename, 'w')
    cPickle.dump({'lpc_filename': prefix + '.' + suffix, 
                  'reader_parameters': reader_parameters, 
                  'lpc_parameters': lpc_parameters}, pk_metadata, 0)
    pk_metadata.close()
    return _metadata_filename
  
class LPCPickler(LPCBaseWriter):
  '''Pickles the lpc curves (actually a dictionary containing the event_id, the points data (perhaps scaled) and lpc_curves) of each event to a file, with filename 
  determined by the target_filename_prefix concatenated with a hash of the lpc parameters. Dumps the parameters used to generate 
  lpc curves to a file that contains a dictionary containing the lpc_parameters and the filename of the pickled object containing 
  all the event's lpc curves.
  '''
  SUFFIX = 'pkl'
  def __init__(self, reader, lpc_algorithm, target_directory, target_filename_prefix):
    LPCBaseWriter.__init__(self, reader, lpc_algorithm, target_directory, target_filename_prefix)
  
  def writeEvent(self, event_id, event):
    '''event_id is ignored for LPCPickler
    '''   
    cPickle.dump(event, self._lpc_data, -1)
class LPCShelver(LPCBaseWriter):
  SUFFIX = 'shl'
  def __init__(self, reader, lpc_algorithm, target_directory, target_filename_prefix):
    LPCBaseWriter.__init__(self, reader, lpc_algorithm, target_directory, target_filename_prefix)
  
  def open(self, prefix):
    '''Overrides the base class to set self._lpc_data as a python shelf 
    '''
    self._lpc_data = shelve.open(prefix + '.shl') 
  
  def writeEvent(self, event_id, event):
    self._lpc_data[str(event_id)] = event

class LPCProcessor(object):
  '''
  Class that processes events (reads parameters from filename, runs the lpc algorithm, then serialises the output 
  '''
  def __init__(self, filename):
    '''
    Constructor
    '''
    self._parser = LPCParameterParser(filename)
    self._initReader()
    self._initStartPoints()
    self._initLpc()
    self._initSerialization()
  
  def _initReader(self):
    run_parameters = self._parser.getRunParameters()
    if run_parameters['type'] == 'LamuRead':
      self._reader = LamuRead(**run_parameters['params'])
    else:
      raise ValueError, 'Specified type of reader is not recognised'
  
  def _initStartPoints(self):
    start_points_parameters = self._parser.getStartPointsParameters()
    if start_points_parameters['type'] == 'lpcMeanShift':
      self._start_points_generator = lpcMeanShift(**start_points_parameters['params'])
    else:
      raise ValueError, 'Specified type of start points generator is not recognised' 
  
  def _initLpc(self):
    lpc_parameters = self._parser.getLpcParameters()
    if lpc_parameters['type'] == 'LPCImpl':
      self._lpcAlgorithm = LPCImpl(self._start_points_generator, **lpc_parameters['params']) 
    else:
      raise ValueError, 'Specified type of lpc algorithm is not recognised'  
  
  def _initSerialization(self):
    serialization_parameters = self._parser.getSerialisationParameters()
    if serialization_parameters['type'] == 'LPCPickler':
      self._writer = LPCPickler(self._reader, self._lpcAlgorithm, **serialization_parameters['params'])
    elif serialization_parameters['type'] == 'LPCShelver':
      self._writer = LPCShelver(self._reader, self._lpcAlgorithm, **serialization_parameters['params'])
    else:
      raise ValueError, 'Specified type of serialization is not recognised'  
  def getMetadataFilename(self):
    return self._writer._metadata_filename
  def runProcessor(self):
    events = self._reader.getEventGenerator()
    lpc = self._lpcAlgorithm
    i = 0
    for event in events:
      lpc_curve = lpc.lpc(X=event.getEventHits())
      lpc_data = {'id': i, 'lpc_curve': lpc_curve, 'Xi': lpc.Xi, 'data_range': lpc._dataRange}
      self._writer.writeEvent(i, lpc_data)
      i += 1
    self._writer.close()
if __name__ == '__main__':
  proc = LPCProcessor('../../resources/test_nonscaled.xml')
  proc.runProcessor()