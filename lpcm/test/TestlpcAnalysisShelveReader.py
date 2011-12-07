from lpcm.lpcAnalysis import lpcAnalysisShelveReader
import pprint
def readShelf(filename):
  reader = lpcAnalysisShelveReader(filename)
  evt = reader.getSingleEvent(1)
  return evt

if __name__ == "__main__":
  evt = readShelf('/tmp/muon-proton_1266297863573875304_meta.pkl')
  pprint.pprint(evt)  
  