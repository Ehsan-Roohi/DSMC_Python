from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from .stage117_pair_node_transition_core import analyze
def run(parent_dir,output_dir):
 p=Path(parent_dir); s=json.loads((p/'summary.json').read_text())
 if s.get('stage')!=116 or s.get('decision')!='stage116_mixed_pair_radial_structure_stage117_pair_node_transition_audit' or s.get('finite') is not True: raise ValueError('parent mismatch')
 c=s['configuration']
 if c.get('grid')!=[64,64] or c.get('kn0')!=10.0 or c.get('rule')!=[40,96] or c.get('pair_sectors')!=[5,6] or any(bool(v) for k,v in c.items() if k.endswith('_retuning')): raise ValueError('fixed design mismatch')
 with np.load(p/'pair_resolved_radial_node_profiles.npz') as d: phi=np.asarray(d['phi'],float); psi=np.asarray(d['psi'],float); speed=np.asarray(d['node_speed_mean'],float)
 m,a,dec=analyze(phi,psi,speed); out={'stage':117,'finite':True,'metrics':m,'aggregate':a,'decision':dec,'scientific_conclusion':'A coherent transition is a localization result only, not a solver-stability or validation result.','design_guard':'No solver endpoint is advanced and no retained setting is changed.'}
 od=Path(output_dir); od.mkdir(parents=True,exist_ok=True); (od/'summary.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); np.savez_compressed(od/'radial_transition_profiles.npz',phi=phi,psi=psi,phi_minus_psi=phi-psi,node_speed_mean=speed); return out
def main():
 a=argparse.ArgumentParser(); a.add_argument('--parent-dir',required=True); a.add_argument('--output-dir',required=True); x=a.parse_args(); run(x.parent_dir,x.output_dir)
if __name__=='__main__': main()
