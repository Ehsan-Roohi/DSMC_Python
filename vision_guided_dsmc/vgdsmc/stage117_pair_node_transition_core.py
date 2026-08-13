from __future__ import annotations
import numpy as np
BANDS=('near_1_4','mid_5_14','inner_15_28')
BOUND=2; MINCOS=.995; MINTV=.30; MAXRR=.10
STABLE='stage117_stable_single_radial_transition_stage118_distribution_role_weighting_audit'
SPATIAL='stage117_wall_distance_dependent_transition_stage118_spatial_transition_audit'
DIFFUSE='stage117_diffuse_radial_transition_stage118_gradient_strength_confound_audit'
def crossings(d):
 return [j for j in range(len(d)-1) if d[j]!=0 and d[j+1]!=0 and d[j]*d[j+1]<0]
def cosine(a,b):
 return float(np.dot(a,b)/max(float(np.linalg.norm(a)*np.linalg.norm(b)),1e-300))
def analyze(phi,psi,speed):
 m={}; diffs=[]; gaps=[]; tvs=[]
 for i,b in enumerate(BANDS):
  p=np.asarray(phi[i],float); q=np.asarray(psi[i],float); p/=p.sum(); q/=q.sum(); d=p-q; diffs.append(d)
  gap=float(np.dot(q-p,speed)); tv=float(.5*np.abs(d).sum()); gaps.append(gap); tvs.append(tv)
  m[b]={'transition_boundaries':crossings(d),'phi_dominant_nodes':[int(x) for x in np.flatnonzero(d>0)],'psi_dominant_nodes':[int(x) for x in np.flatnonzero(d<0)],'centroid_speed_gap_psi_minus_phi':gap,'total_variation_distance':tv,'phi_low_0_2_share':float(p[:3].sum()),'psi_low_0_2_share':float(q[:3].sum())}
 cs=[cosine(diffs[i],diffs[j]) for i in range(3) for j in range(i+1,3)]; g=np.asarray(gaps); rr=float((g.max()-g.min())/abs(g.mean()))
 a={'minimum_cross_band_difference_cosine':min(cs),'centroid_speed_gap_relative_range':rr,'minimum_total_variation':min(tvs),'maximum_total_variation':max(tvs),'mean_total_variation':float(np.mean(tvs)),'mean_centroid_speed_gap':float(np.mean(gaps))}
 same=all(m[b]['transition_boundaries']==[BOUND] for b in BANDS)
 dec=STABLE if same and a['minimum_cross_band_difference_cosine']>=MINCOS and a['minimum_total_variation']>=MINTV and rr<=MAXRR else (SPATIAL if a['minimum_total_variation']>=MINTV and any(len(m[b]['transition_boundaries'])==1 for b in BANDS) else DIFFUSE)
 return m,a,dec
