import cPickle
from collections import defaultdict
import ROOT as rt
def getData(filename):
  f = open(filename, 'rb')
  pkl = cPickle.load(f)
  return pkl
def runEfficiencyPurity(filename, particles):
  p = getData(filename)
  ep_all = []
  for i in range(len(p)):
    r = particleGroupRecode(p[i], particles)
    epr = calcEfficiencyPurity(r)
    ep = calcEfficiencyPurity(p[i])
    ep_all.append({'epr':epr, 'ep':ep})
  return ep_all
def generateHistogram(data, title): 
  rt.gROOT.Reset()
  histo = rt.TH1I("h1",title, 10, 0.95*min(data),1.05*max(data))
  for d in data:
    histo.Fill(d)
  return histo
def plotHistogram(hist):
  c1 = rt.TCanvas("c1","c1",800,800)
  c1.cd(1)
  hist.Draw()
  
def particleGroupRecode(evt_data, particles):
  v = evt_data['voxel_dict']
  recoded_voxels = {}
  for k in v.keys():
    l = v[k]
    recode = set()
    for p in l:
      if p in particles:
        recode.add(1)
      else:
        recode.add(-1)
    recoded_voxels[k] = list(recode)
  
  p = evt_data['pur']
  recoded_pur = []
  for curve in p:
    num_in_particles = 0
    num_not_in_particles = 0  
    for k in curve.keys():
      if k in particles:
        num_in_particles += curve[k]
      else:
        num_not_in_particles += curve[k]
    recode = {}
    if num_in_particles > 0:
      recode[1] = num_in_particles
    if num_not_in_particles > 0:
      recode[-1] = num_not_in_particles
    recoded_pur.append(recode)
    
  return {'voxel_dict': recoded_voxels, 'pur': recoded_pur}
  
def calcEfficiencyPurity(evt):
#filename = './purity_data_1.pkl'
  hits = [p for plist in evt['voxel_dict'].values() for p in plist]
  d = defaultdict(int)
  for h in hits:
    d[h] += 1

  lpc_coverage = evt['pur']

  #calculate efficiency and purity
  evt_eff = {}
  evt_pur = {}
  for p in d.keys():
    p_sum_hits_in_curves = 0
    p_max_hits = 0
    p_max_hits_idx = 0
    for i in range(len(lpc_coverage)):
      try:
        hits = lpc_coverage[i][p] 
        if hits > p_max_hits:
	  p_max_hits = hits
	  p_max_hits_idx = i
        p_sum_hits_in_curves += hits
      except KeyError:
        pass
    p_coverage = float(p_sum_hits_in_curves) / d[p]  
    p_purity = float(p_max_hits)/sum(lpc_coverage[p_max_hits_idx].values())  
    evt_eff[p] = p_coverage
    evt_pur[p] = p_purity
  
  eff_pur = {'eff':evt_eff, 'pur': evt_pur, 'num_curves': len(lpc_coverage)}
  return eff_pur
