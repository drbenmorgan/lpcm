from lpcm.lpcAnalysis import lpcAnalysisShelveReader
import pprint
def readShelf(filename):
  reader = lpcAnalysisShelveReader(filename)
  evt = reader.getSingleEvent(1)
  return evt
def readShelfGenerator(filename):
  reader = lpcAnalysisShelveReader(filename)
  evt = reader.getNextEvent()
  return evt
if __name__ == "__main__":
  #test the index based lpcProcessing output reader
  evt = readShelf('/tmp/muon-proton_1266297863573875304_meta.pkl')
  pprint.pprint(evt)  
  #test the pickle-style generator next event function
  evt = readShelfGenerator('/tmp/muon-proton_1266297863573875304_meta.pkl')
  pprint.pprint(evt)