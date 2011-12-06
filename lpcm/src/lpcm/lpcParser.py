'''
Created on 6 Dec 2011

@author: droythorne
'''
from elementtree.ElementTree import ElementTree

class lpcParameterParser(object):
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
    parser_type_tag = self.__class__.TYPE_TAG
    parser_type_node = self._config_tree.getiterator(parser_type_tag)
    if len(parser_type_node) == 1:
      param_node = parser_type_node[0].getiterator(tag)
      if len(param_node) == 1:
        param_type = param_node[0].get('type')
        params = {}
        for par in param_node[0]:  
          items = dict(par.items())
          s = 'v=' + items['type'] + '("' + items['value'] + '")' 
          exec(s) #TODO - remove exec, there are clearly better ways to do this!
          params[items['name']] = v
        return {'type': param_type, 'params': params}
      else:
        msg = 'The required unique lpc configuration element tag, ' + parser_type_tag + ' is missing or not unique from ' + self._filename
        raise ValueError, msg 
    else:
      msg = 'The required unique lpc configuration element tag, ' + parser_type_tag + ' is missing or not unique from ' + self._filename
      raise ValueError, msg
    
class lpcProcessingParser(lpcParameterParser):  
  TYPE_TAG = 'lpcProcessing'
  
  def __init__(self, filename):
    lpcParameterParser.__init__(self, filename) 
  
  def getRunParameters(self):
    '''Gets the filename of event file, type that reads it in and, if present, max number of 
    events to process (set as None if absent)
    'max_events', 'filename'
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
   
class lpcAnalysisParser(lpcParameterParser):  
  TYPE_TAG = 'lpcAnalysis'
  
  def __init__(self, filename):
    lpcParameterParser.__init__(self, filename) 
  
  def getReadParameters(self):
    '''Gets the filename of processing metadata file, type that reads it in and, if present, max number of 
    events to analyse (set as None if absent)
    'max_events', 'metadata_filename'
    '''
    d = self._generateParamDictionary('Read')
    return d
  def getResidualsParameters(self):
    d = self._generateParamDictionary('Residuals')
    return d