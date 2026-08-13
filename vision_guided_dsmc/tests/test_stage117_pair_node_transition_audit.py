import json
import numpy as np
from vgdsmc.stage117_pair_node_transition_core import STABLE,analyze,crossings
from vgdsmc.stage117_pair_node_transition_audit import run
def profiles():
 p=np.tile(np.array([.20,.18,.12,.08,.10,.10,.08,.06,.04,.02]),(3,1)); q=np.tile(np.array([.02,.04,.08,.14,.17,.17,.15,.11,.08,.04]),(3,1)); return p,q,np.linspace(.4,1.8,10)
def test_crossing(): assert crossings(np.array([2.,1.,.1,-.1,-1.]))==[2]
def test_stable_core():
 p,q,s=profiles(); m,a,d=analyze(p,q,s); assert d==STABLE; assert all(v['transition_boundaries']==[2] for v in m.values()); assert a['minimum_cross_band_difference_cosine']>.999
def test_runner(tmp_path):
 p,q,s=profiles(); parent=tmp_path/'p'; out=tmp_path/'o'; parent.mkdir(); (parent/'summary.json').write_text(json.dumps({'stage':116,'decision':'stage116_mixed_pair_radial_structure_stage117_pair_node_transition_audit','finite':True,'configuration':{'grid':[64,64],'kn0':10.0,'rule':[40,96],'pair_sectors':[5,6],'physical_parameter_retuning':False}})); np.savez_compressed(parent/'pair_resolved_radial_node_profiles.npz',phi=p,psi=q,node_speed_mean=s); z=run(parent,out); assert z['stage']==117 and z['decision']==STABLE; assert (out/'radial_transition_profiles.npz').exists()
