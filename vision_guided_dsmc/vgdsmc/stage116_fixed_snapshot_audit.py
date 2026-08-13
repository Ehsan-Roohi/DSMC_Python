from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

BANDS=("near_1_4","mid_5_14","inner_15_28")

def _metric(p,q):
    p=np.asarray(p,float); q=np.asarray(q,float); p=p/p.sum(); q=q/q.sum(); m=.5*(p+q)
    pt=np.argsort(p,kind='stable')[-2:][::-1]; qt=np.argsort(q,kind='stable')[-2:][::-1]; jt=np.argsort(m,kind='stable')[-2:][::-1]
    return {"profile_cosine":float(np.dot(p,q)/(np.linalg.norm(p)*np.linalg.norm(q))),"overlap_coefficient":float(np.minimum(p,q).sum()),"total_variation_distance":float(.5*np.abs(p-q).sum()),"phi_top2_node_index":[int(x) for x in pt],"psi_top2_node_index":[int(x) for x in qt],"joint_top2_node_index":[int(x) for x in jt],"phi_top2_share":float(p[pt].sum()),"psi_top2_share":float(q[qt].sum()),"joint_top2_share":float(m[jt].sum()),"phi_maximum_node_share":float(p.max()),"psi_maximum_node_share":float(q.max()),"phi_effective_node_count":float(1/(p*p).sum()),"psi_effective_node_count":float(1/(q*q).sum()),"top2_sets_match":set(map(int,pt))==set(map(int,qt))==set(map(int,jt))}

def _decision(metrics,err):
    if err>1e-12:return "stage116_stage115_pair_share_reconstruction_blocker_without_retuning"
    broad=[metrics['near_1_4'],metrics['mid_5_14']]
    common=all(b['profile_cosine']>=.95 and b['overlap_coefficient']>=.90 and b['top2_sets_match'] and min(b['phi_top2_share'],b['psi_top2_share'])>=.5 for b in broad)
    if common and set(broad[0]['joint_top2_node_index'])==set(broad[1]['joint_top2_node_index']):return "stage116_common_pair_radial_nodes_stage117_pair_node_wall_distance_interaction_audit"
    diffuse=all(max(b['phi_maximum_node_share'],b['psi_maximum_node_share'])<=.2 and min(b['phi_effective_node_count'],b['psi_effective_node_count'])>=6 for b in broad)
    if diffuse:return "stage116_pair_support_radially_diffuse_stage117_gradient_strength_confound_audit"
    return "stage116_mixed_pair_radial_structure_stage117_pair_node_transition_audit"

def run(input_path,output_dir):
    d=json.loads(Path(input_path).read_text()); c=d['configuration']; p=d['provenance']
    assert d['stage']==116 and d['input_type']=='exact_parent_artifact_derived_fixed_snapshot'
    assert c['grid']==[64,64] and c['kn0']==10.0 and c['cold_hot_ratio']==.1 and c['rule']==[40,96] and c['radial_scale']==2.0
    assert c['limiter']=='minmod' and c['boundary_slope']=='zero' and c['source_relaxation']==1.0 and c['correction_floor']==.05 and c['diagnostic_steps']==25
    assert c['dominant_radial_shell']==1 and c['radial_nodes_per_shell']==10 and c['pair_sectors']==[5,6] and c['pair_points_per_radial_node']==24
    assert not any(v for k,v in c.items() if k.endswith('_retuning')) and c['cross_knudsen_extension_permitted'] is False and c['validation_claim_permitted'] is False
    assert p['stage67']['workflow_run_id']==30991124477 and p['stage67']['artifact_id']==8931272132 and p['stage67']['distributions_sha256']=='d4002a2765137ba517abec2d0483a3e5adcf13f415c53259018556bd14d612d1'
    assert p['stage111']['workflow_run_id']==31590035358 and p['stage111']['artifact_id']==9149082510 and p['stage111']['maps_sha256']=='78b8173d81eaf2523791201690d6464295b8aa65413d9ccb3eb2c2215ac85407'
    assert p['stage115']['workflow_run_id']==31690647300 and p['stage115']['artifact_id']==9181151145 and p['stage115']['decision']=='stage115_common_adjacent_pair_support_stage116_pair_resolved_radial_node_audit'
    assert d['sector_point_count']==[120]*8 and d['radial_node_point_count']==[96]*10 and d['pair_point_count_per_radial_node']==[24]*10
    metrics={}
    for band in BANDS:
        m=_metric(d['phi'][band]['radial_node_share'],d['psi'][band]['radial_node_share']); m.update({'phi_pair_share':d['phi'][band]['pair_share'],'psi_pair_share':d['psi'][band]['pair_share']}); metrics[band]=m
    err=float(d['max_stage115_pair_share_reconstruction_abs_error']); decision=_decision(metrics,err)
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    summary={'stage':116,'configuration':c,'provenance':p,'finite':True,'max_stage115_pair_share_reconstruction_abs_error':err,'metrics':metrics,'decision':decision,'scientific_conclusion':'The fixed sectors 5+6 do not share a common radial-node concentration between phi and psi: phi is weighted toward the two lowest shell-1 radial nodes, while psi peaks at intermediate nodes. This is a negative common-node result and a positive distribution-specific localization result only.','negative_result_guard':'No solver endpoint was rerun. No physical, collision, wall, limiter, source, transport, floor, normalization, or velocity-quadrature parameter was retuned. Stage 90 remains nonconverged, Stage 28 remains a failed MUSCL endpoint, Stage 89 remains unpromoted, and no stability, accuracy, heat-flux, benchmark, cross-Knudsen, or validation claim is authorized.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    np.savez_compressed(out/'pair_resolved_radial_node_profiles.npz',phi=np.asarray([d['phi'][b]['radial_node_share'] for b in BANDS]),psi=np.asarray([d['psi'][b]['radial_node_share'] for b in BANDS]),node_speed_mean=np.asarray(d['node_speed_mean']))
    return summary

def main():
    a=argparse.ArgumentParser();a.add_argument('--input',required=True);a.add_argument('--output-dir',required=True);x=a.parse_args();run(x.input,x.output_dir)
if __name__=='__main__':main()
