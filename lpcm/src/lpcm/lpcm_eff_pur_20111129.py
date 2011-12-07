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
  recoded_pur = {}
  for tau in p.iterkeys():
    recoded_pur_tau = []
    for curve in p[tau]:
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
      recoded_pur_tau.append(recode)
    recoded_pur[tau] = recoded_pur_tau
    
  return {'voxel_dict': recoded_voxels, 'pur': recoded_pur}
  
def calcEfficiencyPurity(evt):
#filename = './purity_data_1.pkl'
  hits = [p for plist in evt['voxel_dict'].values() for p in plist]
  d = defaultdict(int)
  for h in hits:
    d[h] += 1

  lpc_coverage = evt['pur']
  eff_pur_dict = {}
  for tau in lpc_coverage.iterkeys():
  #calculate efficiency and purity
    evt_eff = {}
    evt_pur = {}
    for p in d.keys():
      p_sum_hits_in_curves = 0
      p_max_hits = 0
      p_max_hits_idx = 0
      for i in range(len(lpc_coverage[tau])):
        try:
          hits = lpc_coverage[tau][i][p] 
          if hits > p_max_hits:
  	         p_max_hits = hits
  	         p_max_hits_idx = i
          p_sum_hits_in_curves += hits
        except KeyError:
          pass
      p_coverage = float(p_sum_hits_in_curves) / d[p]  
      p_purity = float(p_max_hits)/sum(lpc_coverage[tau][p_max_hits_idx].values())  
      evt_eff[p] = p_coverage
      evt_pur[p] = p_purity
    
    eff_pur = {'eff':evt_eff, 'pur': evt_pur, 'num_curves': len(lpc_coverage[tau])}
    eff_pur_dict[tau] = eff_pur
  return eff_pur_dict
'''
import lpcm.lpcm_eff_pur_20111129

#read in the output from an lpcAnalysis run and calcualte the efficiencies and purities based on individual particle ids
#and the combination of pdg codes 2212 and 13 (proton and muon), pureff is a list (indexed by event number) of dictionaries, with 
#keys 'epr' ('binary' efficiency/purity which groups pdg codes in 'particles' argument to runEfficiencyPurity) and 'ep' (which
#gives efficiency/purity for each particle individually). The values of each of these keys are dictionaries with keys corresponding
#the tau parameters given to lpcAnalysis (tau is the radius of segment-wise cylinders around a track use to associate a hit with
#the given track). Each value is a dictionary with keys 'pur', the purity* for each particle or group of particles (+1 for particles
#in 'particles' argument, -1 for the other particles); 'eff' the efficiency (what proportion of all particles in the event of a 
#given type are within tau of a constructed lpc curve) 
# *purity for a particular particle, p,  is defined as the the proportion of particle p hits (p-hits) of all hits within tau 
#of the curve that has the largest number of p-hits within tau. For voxels that contain hits from multiple particles, every particle
#counts towards purity and efficiency (i.e. in the proton purity calculation, voxels with, say, a proton AND electron hit will count twice (once for each particle) 
#in the denominator and only once in the numerator; for the proton efficiency, the proton hit will be included in numerator and denominator; 
#so it might be said that purity is incomplete in that it only picks out the curve with the most associated hits, but pessimistic
#in that it counts multi-particle voxel hits as impurities, so that simultaneous 100% efficiency and 100% purity is impossible 
 
pureff = pe.runEfficiencyPurity('/home/droythorne/git/physics/lpcm/lpcm/resources/purity_data.pkl', [2212,13])
pureff[0]['epr']
