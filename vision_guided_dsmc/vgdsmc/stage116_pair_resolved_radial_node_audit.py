from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from . import stage110_same_sign_slope_asymmetry_audit as s110
from . import stage114_wall_distance_conditioned_velocity_quadrature_audit as s114

STAGE115_RUN_ID=31690647300
STAGE115_JOB_ID=94416925424
STAGE115_ARTIFACT_ID=9181151145
STAGE115_ARTIFACT_SHA256="8245baedf9ad29db6c8b0d290e8507069479c867f449fc67d2b215a65edef3a1"
STAGE115_SUMMARY_SHA256="9c36830cc56cae1745075f57a0a0992656e09486bf5c90a93eb93e7f251b30e6"
STAGE115_PROFILES_SHA256="932a643786f94b08c6156fbf985d9c39b094f98532a2480190a4ded791bcf02a"
STAGE115_SOURCE_HEAD="2d189102c0bca08c1f9d4a5a56daedd482fcd914"
STAGE115_DECISION="stage115_common_adjacent_pair_support_stage116_pair_resolved_radial_node_audit"
GRID=s114.GRID; KNUDSEN=s114.KNUDSEN; COLD_HOT_RATIO=s114.COLD_HOT_RATIO
RULE=s114.RULE; RADIAL_SCALE=s114.RADIAL_SCALE; LIMITER=s114.LIMITER
BOUNDARY_SLOPE=s114.BOUNDARY_SLOPE; SOURCE_RELAXATION=s114.SOURCE_RELAXATION
TOLERANCE=s114.TOLERANCE; CORRECTION_FLOOR=s114.CORRECTION_FLOOR
DIAGNOSTIC_STEPS=s114.DIAGNOSTIC_STEPS; WALL_BAND_CELLS=s114.WALL_BAND_CELLS
DOMINANT_RADIAL_SHELL=s114.DOMINANT_RADIAL_SHELL; RADIAL_SHELL_COUNT=s114.RADIAL_SHELL_COUNT
RADIAL_NODES_PER_SHELL=s114.RADIAL_NODES_PER_SHELL; ANGULAR_SECTORS=s114.ANGULAR_SECTORS
NEAR_WALL_DEPTH=s114.NEAR_WALL_DEPTH; BROAD_WALL_DEPTH=s114.BROAD_WALL_DEPTH
PAIR_SECTORS=(5,6); BANDS=("near_1_4","mid_5_14","inner_15_28")
PAIR_SHARE_RECONSTRUCTION_TOLERANCE=1e-12
NODE_PROFILE_COSINE_COMMON_MIN=.95; NODE_PROFILE_OVERLAP_COMMON_MIN=.90
COMMON_TOP2_NODE_SHARE_MIN=.50; DIFFUSE_MAX_NODE_SHARE=.20; DIFFUSE_EFFECTIVE_NODE_MIN=6.0


def validate_stage116_design(**overrides: object)->None:
    frozen={"grid":GRID,"kn0":KNUDSEN,"cold_hot_ratio":COLD_HOT_RATIO,"rule":RULE,
      "radial_scale":RADIAL_SCALE,"limiter":LIMITER,"boundary_slope":BOUNDARY_SLOPE,
      "source_relaxation":SOURCE_RELAXATION,"tolerance":TOLERANCE,"correction_floor":CORRECTION_FLOOR,
      "diagnostic_steps":DIAGNOSTIC_STEPS,"wall_band_cells":WALL_BAND_CELLS,
      "dominant_radial_shell":DOMINANT_RADIAL_SHELL,"radial_shell_count":RADIAL_SHELL_COUNT,
      "radial_nodes_per_shell":RADIAL_NODES_PER_SHELL,"angular_sectors":ANGULAR_SECTORS,
      "pair_sectors":PAIR_SECTORS,"stage67_run_id":s110.STAGE67_RUN_ID,
      "stage111_run_id":s114.STAGE111_RUN_ID,"stage115_run_id":STAGE115_RUN_ID}
    if any(k not in frozen or frozen[k]!=v for k,v in overrides.items()):
        raise ValueError("Stage 116 is frozen; no physical, quadrature, limiter, wall, source, transport, floor, or failed MUSCL parameter may be retuned")
    if RULE!=(40,96) or RADIAL_NODES_PER_SHELL!=10 or PAIR_SECTORS!=(5,6):
        raise ValueError("Stage 116 requires the exact 40x96 rule, ten shell-1 radial nodes, and sectors 5+6")


def _load_stage115_record(path:str|Path)->dict[str,object]:
    r=json.loads(Path(path).read_text())
    checks=(r.get("stage")==115,r.get("decision")==STAGE115_DECISION,r.get("finite") is True,
      r.get("source_head")==STAGE115_SOURCE_HEAD,r.get("workflow_status")=="completed",
      r.get("workflow_conclusion")=="success",r.get("workflow_run_id")==STAGE115_RUN_ID,
      r.get("workflow_job_id")==STAGE115_JOB_ID,r.get("artifact_id")==STAGE115_ARTIFACT_ID,
      r.get("artifact_sha256")==STAGE115_ARTIFACT_SHA256,r.get("summary_sha256")==STAGE115_SUMMARY_SHA256,
      r.get("distribution_specific_sector_profiles_sha256")==STAGE115_PROFILES_SHA256,
      r.get("tests",{}).get("passed")==9,r.get("tests",{}).get("failed")==0)
    if not all(checks): raise ValueError("Committed Stage-115 provenance does not authorize Stage 116")
    for band in ("near_1_4","mid_5_14"):
        b=r["metrics"][band]
        if b["joint_top2_sector_index"]!=[5,6] or b["phi_top2_sector_index"]!=[5,6] or b["psi_top2_sector_index"]!=[5,6]:
            raise ValueError("Stage-115 common pair is not frozen sectors 5+6")
        if min(float(b["phi_top2_share"]),float(b["psi_top2_share"]))<.5:
            raise ValueError("Stage-115 pair support does not satisfy the fixed guard")
    return r


def radial_node_indices_within_shell(vx:np.ndarray,vy:np.ndarray)->np.ndarray:
    speed=np.hypot(vx,vy)
    if speed.shape!=(960,) or not np.isfinite(speed).all(): raise ValueError("Stage 116 requires exact 960-point shell-1 support")
    order=np.argsort(speed,kind="stable"); labels=np.empty(960,dtype=np.int16)
    labels[order]=np.repeat(np.arange(10,dtype=np.int16),96)
    for j in range(10):
        q=speed[labels==j]
        if q.size!=96 or float(q.max()-q.min())>1e-12*max(float(q.mean()),1.0): raise ValueError("Radial-node grouping mixed distinct speeds")
    return labels


def _x_same_sign_change_pointwise(f:np.ndarray)->np.ndarray:
    f=np.asarray(f,dtype=float); w=WALL_BAND_CELLS
    c=f[w:-w,w:-w]; left=c-f[w:-w,w-1:-w-1]; right=f[w:-w,w+1:-w+1]-c
    same=((left>0)&(right>0))|((left<0)&(right<0))
    return np.where(same,.5*np.abs(np.abs(left)-np.abs(right)),0.0)


def _profile_metrics(phi:np.ndarray,psi:np.ndarray)->dict[str,object]:
    p=np.asarray(phi,dtype=float); q=np.asarray(psi,dtype=float)
    p=p/max(float(p.sum()),1e-300); q=q/max(float(q.sum()),1e-300); m=.5*(p+q)
    pt=np.argsort(p,kind="stable")[-2:][::-1]; qt=np.argsort(q,kind="stable")[-2:][::-1]; jt=np.argsort(m,kind="stable")[-2:][::-1]
    return {"profile_cosine":float(np.dot(p,q)/max(float(np.linalg.norm(p)*np.linalg.norm(q)),1e-300)),
      "overlap_coefficient":float(np.minimum(p,q).sum()),"total_variation_distance":float(.5*np.abs(p-q).sum()),
      "phi_top2_node_index":[int(x) for x in pt],"psi_top2_node_index":[int(x) for x in qt],"joint_top2_node_index":[int(x) for x in jt],
      "phi_top2_share":float(p[pt].sum()),"psi_top2_share":float(q[qt].sum()),"joint_top2_share":float(m[jt].sum()),
      "top2_sets_match":set(map(int,pt))==set(map(int,qt))==set(map(int,jt))}


def stage116_decision(metrics:dict[str,dict[str,object]],finite:bool,max_pair_share_error:float)->str:
    if not finite or not np.isfinite(max_pair_share_error): return "stage116_nonfinite_pair_radial_node_blocker_without_retuning"
    if max_pair_share_error>PAIR_SHARE_RECONSTRUCTION_TOLERANCE: return "stage116_stage115_pair_share_reconstruction_blocker_without_retuning"
    broad=[metrics["near_1_4"],metrics["mid_5_14"]]
    common=all(float(b["profile_cosine"])>=.95 and float(b["overlap_coefficient"])>=.90 and bool(b["top2_sets_match"]) and min(float(b["phi_top2_share"]),float(b["psi_top2_share"]))>=.50 for b in broad)
    if common and set(broad[0]["joint_top2_node_index"])==set(broad[1]["joint_top2_node_index"]): return "stage116_common_pair_radial_nodes_stage117_pair_node_wall_distance_interaction_audit"
    diffuse=all(max(float(b["phi_maximum_node_share"]),float(b["psi_maximum_node_share"]))<=.20 and min(float(b["phi_effective_node_count"]),float(b["psi_effective_node_count"]))>=6 for b in broad)
    if diffuse: return "stage116_pair_support_radially_diffuse_stage117_gradient_strength_confound_audit"
    return "stage116_mixed_pair_radial_structure_stage117_pair_node_transition_audit"


def run_stage116(stage67_artifact_dir:str|Path,stage111_artifact_dir:str|Path,stage115_record_path:str|Path,output_dir:str|Path,**design:object)->dict[str,object]:
    validate_stage116_design(**design); parent=_load_stage115_record(stage115_record_path)
    s67,distfile=s110._load_stage67(stage67_artifact_dir); s111,maps=s114._load_stage111(stage111_artifact_dir)
    prof={"phi":[],"psi":[]}; sectorprof={"phi":[],"psi":[]}; pairshare={"phi":[],"psi":[]}; errors=[]
    with np.load(distfile) as d:
        vx=np.asarray(d["vx"]); vy=np.asarray(d["vy"]); weight=np.asarray(d["weight"])
        shell=s110._radial_shell_indices(vx,vy)==DOMINANT_RADIAL_SHELL
        svx=vx[shell]; svy=vy[shell]; sw=weight[shell]; sectors=s114.angular_sector_indices(svx,svy); nodes=radial_node_indices_within_shell(svx,svy)
        pair=np.isin(sectors,PAIR_SECTORS)
        if any(np.count_nonzero((nodes==j)&pair)!=24 for j in range(10)): raise ValueError("Each radial node must contain 24 points in sectors 5+6")
        speed=np.hypot(svx,svy); nsmean=np.array([speed[nodes==j].mean() for j in range(10)]); nsmin=np.array([speed[nodes==j].min() for j in range(10)]); nsmax=np.array([speed[nodes==j].max() for j in range(10)])
        bands=s114.wall_distance_band_masks()
        for name in ("phi","psi"):
            point=_x_same_sign_change_pointwise(np.asarray(d[name])[...,shell]); weighted=point*sw[None,None,:]*maps[f"{name}_growth_amplitude"][...,None]
            for band in BANDS:
                cells=weighted[bands[band]]; total=float(cells.sum())
                raw=np.array([cells[:,(nodes==j)&pair].sum() for j in range(10)],dtype=float)
                bysector=np.array([[cells[:,(nodes==j)&(sectors==k)].sum() for j in range(10)] for k in PAIR_SECTORS],dtype=float)
                ps=float(raw.sum()/max(total,1e-300)); expected=float(parent["metrics"][band][f"{name}_top2_share"]); errors.append(abs(ps-expected))
                prof[name].append(raw/max(float(raw.sum()),1e-300)); sectorprof[name].append(bysector/max(float(raw.sum()),1e-300)); pairshare[name].append(ps)
    metrics={}
    for i,band in enumerate(BANDS):
        p=np.asarray(prof["phi"][i]); q=np.asarray(prof["psi"][i]); b=_profile_metrics(p,q)
        b.update({"phi_pair_share":float(pairshare["phi"][i]),"psi_pair_share":float(pairshare["psi"][i]),"phi_radial_node_share":p.tolist(),"psi_radial_node_share":q.tolist(),
          "phi_maximum_node_index":int(p.argmax()),"psi_maximum_node_index":int(q.argmax()),"phi_maximum_node_share":float(p.max()),"psi_maximum_node_share":float(q.max()),
          "phi_effective_node_count":float(1/(p*p).sum()),"psi_effective_node_count":float(1/(q*q).sum())}); metrics[band]=b
    maxerr=float(max(errors)); finite=np.isfinite(maxerr) and all(np.isfinite(np.asarray(v)).all() for v in prof.values()); decision=stage116_decision(metrics,bool(finite),maxerr)
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/"pair_resolved_radial_node_profiles.npz",phi_radial_node_share=np.asarray(prof["phi"]),psi_radial_node_share=np.asarray(prof["psi"]),phi_sector_node_share=np.asarray(sectorprof["phi"]),psi_sector_node_share=np.asarray(sectorprof["psi"]),phi_pair_share=np.asarray(pairshare["phi"]),psi_pair_share=np.asarray(pairshare["psi"]),node_speed_mean=nsmean,node_speed_min=nsmin,node_speed_max=nsmax,shell_sector_index=sectors,shell_radial_node_index=nodes,shell_weight=sw)
    cfg={"grid":list(GRID),"kn0":KNUDSEN,"cold_hot_ratio":COLD_HOT_RATIO,"rule":list(RULE),"radial_scale":RADIAL_SCALE,"limiter":LIMITER,"boundary_slope":BOUNDARY_SLOPE,"source_relaxation":SOURCE_RELAXATION,"tolerance":TOLERANCE,"correction_floor":CORRECTION_FLOOR,"diagnostic_steps":DIAGNOSTIC_STEPS,"wall_band_cells":WALL_BAND_CELLS,"dominant_radial_shell":DOMINANT_RADIAL_SHELL,"radial_shell_count":RADIAL_SHELL_COUNT,"radial_nodes_per_shell":RADIAL_NODES_PER_SHELL,"angular_sectors":ANGULAR_SECTORS,"pair_sectors":[5,6],"pair_points_per_radial_node":24,"stage67_run_id":s110.STAGE67_RUN_ID,"stage111_run_id":s114.STAGE111_RUN_ID,"stage115_run_id":STAGE115_RUN_ID,"full_solver_endpoint_rerun":False,"physical_parameter_retuning":False,"collision_parameter_retuning":False,"correction_floor_retuning":False,"positivity_floor_retuning":False,"source_relaxation_retuning":False,"transport_parameter_retuning":False,"wall_model_retuning":False,"normalization_retuning":False,"limiter_retuning":False,"velocity_quadrature_retuning":False,"failed_muscl_endpoint_rehabilitated":False,"one_sided_boundary_slope_promoted":False,"cross_knudsen_extension_permitted":False,"validation_claim_permitted":False,"solver_endpoint_claim_permitted":False}
    summary={"stage":116,"configuration":cfg,"stage67_authorization":{"stage":s67["stage"],"decision":s67["decision"]},"stage111_authorization":{"decision":s111["decision"],"workflow_run_id":s114.STAGE111_RUN_ID,"artifact_id":s114.STAGE111_ARTIFACT_ID},"stage115_authorization":{"decision":parent["decision"],"workflow_run_id":STAGE115_RUN_ID,"workflow_job_id":STAGE115_JOB_ID,"artifact_id":STAGE115_ARTIFACT_ID,"tests_passed":9,"tests_failed":0},"finite":bool(finite),"max_stage115_pair_share_reconstruction_abs_error":maxerr,"metrics":metrics,"decision":decision,
      "scientific_conclusion":"This fixed audit only asks whether Stage-115 sectors 5+6 concentrate on the same radial nodes for phi and psi; it does not justify changing quadrature or any failed parameter.",
      "negative_result_guard":"Stage 116 is attribution, not a solver experiment. Stage 115 establishes only shared adjacent-pair support; Stage 111 remains association rather than causality; Stage 90 remains nonconverged; Stage 28 remains a failed MUSCL endpoint; Stage 89 remains unpromoted. No failed parameter or velocity quadrature is retuned, no cross-Knudsen extension is permitted, and no stability, accuracy, benchmark, heat-flux-improvement, or validation claim is authorized."}
    (out/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n"); return summary


def main()->None:
    p=argparse.ArgumentParser(); p.add_argument("--stage67-artifact-dir",required=True); p.add_argument("--stage111-artifact-dir",required=True); p.add_argument("--stage115-record-path",required=True); p.add_argument("--output-dir",required=True); a=p.parse_args()
    run_stage116(a.stage67_artifact_dir,a.stage111_artifact_dir,a.stage115_record_path,a.output_dir)

if __name__=="__main__": main()
