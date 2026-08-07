from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path
import numpy as np

STAGE82_COMPLETED_ENDPOINT = {
    "workflow_run_id": 31203808127,
    "workflow_job_id": 92949872247,
    "workflow_conclusion": "success",
    "tests_passed": 257,
    "tests_failed": 0,
    "artifact_id": 9011460708,
    "artifact_sha256": "4149406d62861d88df727861960505968c4fd383317a4a880662797b938b57dd",
    "source_head_sha": "8401cb19d4f20ef0b34ccf8129569d102db80eb3",
    "summary_sha256": "4c455deff95fe84a5466ea6d7cd9c4b30afb946c68125d7ac867486f9f0fd269",
    "maps_sha256": "aece8100ee64fc0d9937efd8370a92f3d1a147561185d7f1588ff4b13a217765",
    "decision": "stage82_smooth_retained_vertical_oblique_sectors_stage83_opposite_sector_spatial_cancellation_audit",
}
GRID=(64,64); KNUDSEN=10.0; COLD_HOT_RATIO=0.1; RADIAL_NODES=40; ANGULAR_NODES=96
POINT_COUNT=3840; RADIAL_SCALE=2.0; LIMITER="minmod"; DOMINANT_MOMENT="transverse_kinetic"
DOMINANT_RADIAL_SHELL=2; DOMINANT_LOCAL_RADIAL_NODE=1; DOMINANT_GLOBAL_RADIAL_NODE=21
VERTICAL_OBLIQUE_BINS=(1,2,5,6); OPPOSITE_SECTOR_PAIRS=((1,5),(2,6))
OUTER_X_QUARTER_WIDTH=16; SIDE_WALL_EIGHTH_WIDTH=8
PAIR_RETENTION_GUARD=.75; ROWWISE_SIGNED_RETENTION_GUARD=.10; OUTER_X_LOCALIZATION_GUARD=.75
FACE_TO_CELL_CANCELLATION_GUARD=.20; CLOSURE_GUARD=1e-10


def validate_stage83_design(**overrides):
    frozen={"grid":GRID,"kn0":KNUDSEN,"cold_hot_ratio":COLD_HOT_RATIO,"radial_nodes":RADIAL_NODES,
            "angular_nodes":ANGULAR_NODES,"radial_scale":RADIAL_SCALE,"limiter":LIMITER,
            "vertical_oblique_bins":VERTICAL_OBLIQUE_BINS,"opposite_sector_pairs":OPPOSITE_SECTOR_PAIRS,
            "outer_x_quarter_width":OUTER_X_QUARTER_WIDTH,"side_wall_eighth_width":SIDE_WALL_EIGHTH_WIDTH}
    for k,v in overrides.items():
        if k not in frozen or v != frozen[k]:
            raise ValueError("Stage 83 is frozen to the completed Stage-82 endpoint; no solver or parameter retuning is permitted")


def _sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest()


def _validate_stage82_artifact(root):
    root=Path(root); summary=root/"summary.json"; maps=root/"within_sector_angular_coherence_maps.npz"
    if not summary.exists() or not maps.exists(): raise FileNotFoundError("Stage 83 requires exact Stage-82 summary and maps")
    if _sha256(summary)!=STAGE82_COMPLETED_ENDPOINT["summary_sha256"]: raise ValueError("Stage-82 summary checksum mismatch")
    if _sha256(maps)!=STAGE82_COMPLETED_ENDPOINT["maps_sha256"]: raise ValueError("Stage-82 maps checksum mismatch")
    s=json.loads(summary.read_text())
    if s.get("stage")!=82 or s.get("decision")!=STAGE82_COMPLETED_ENDPOINT["decision"]: raise ValueError("Stage-82 decision mismatch")
    if not s.get("finite") or not s.get("closure_closed"): raise ValueError("Stage-82 endpoint is not finite and closure-closed")
    m=s["within_sector_metrics"]
    if m["minimum_sector_cell_weighted_adjacent_coherence"]<.90 or m["minimum_sector_cell_internal_retention_ratio"]<.75:
        raise ValueError("Stage-82 decision branch does not authorize Stage 83")
    c=s["configuration"]
    expected={"grid":[64,64],"kn0":10.0,"radial_nodes":40,"angular_nodes":96,"point_count":3840,
              "radial_scale":2.0,"limiter":"minmod","dominant_moment":DOMINANT_MOMENT,
              "dominant_radial_shell":2,"dominant_local_radial_node":1,"dominant_global_radial_node":21,
              "vertical_oblique_bins":[1,2,5,6],"opposite_sector_pairs":[[1,5],[2,6]],"solver_rerun_count":0,
              "failed_muscl_endpoint_rehabilitated":False,"cross_knudsen_extension_permitted":False,
              "validation_claim_permitted":False}
    for k,v in expected.items():
        if c.get(k)!=v: raise ValueError(f"Stage-82 frozen configuration mismatch for {k}")
    if any(v is not False for k,v in c.items() if k.endswith("_retuning")): raise ValueError("Stage-82 contains retuning")
    return s


def _retention(a):
    d=float(np.sum(np.abs(a))); return 0.0 if d<=1e-300 else float(abs(np.sum(a))/d)


def _profile_retention(a,axis):
    d=float(np.sum(np.abs(a))); return 0.0 if d<=1e-300 else float(np.sum(np.abs(np.sum(a,axis=axis)))/d)


def _rel_l2(a,b):
    d=float(np.linalg.norm(np.ravel(b))); return 0.0 if d<=1e-300 else float(np.linalg.norm(np.ravel(a-b))/d)


def spatial_cancellation_metrics(face,cell,bins):
    face=np.asarray(face,float); cell=np.asarray(cell,float); bins=np.asarray(bins,np.int16)
    if face.shape!=(4,64,63) or cell.shape!=(4,64,64): raise ValueError("Stage 83 requires exact Stage-82 sector-map shapes")
    if not np.array_equal(bins,[1,2,5,6]): raise ValueError("Stage 83 requires exact inherited vertical-oblique labels")
    if not np.all(np.isfinite(face)) or not np.all(np.isfinite(cell)): raise ValueError("Stage 83 requires finite maps")
    idx={int(b):i for i,b in enumerate(bins)}; pf=[]; pc=[]; rows=[]
    for a,b in OPPOSITE_SECTOR_PAIRS:
        i,j=idx[a],idx[b]; f=face[i]+face[j]; c=cell[i]+cell[j]; pf.append(f); pc.append(c)
        fa=float(np.sum(np.abs(f))); ca=float(np.sum(np.abs(c)))
        fd=float(np.sum(np.abs(face[i]))+np.sum(np.abs(face[j]))); cd=float(np.sum(np.abs(cell[i]))+np.sum(np.abs(cell[j])))
        q=OUTER_X_QUARTER_WIDTH; e=SIDE_WALL_EIGHTH_WIDTH
        outer=float(np.sum(np.abs(c[:,:q]))+np.sum(np.abs(c[:,-q:])))
        side=float(np.sum(np.abs(c[:,:e]))+np.sum(np.abs(c[:,-e:])))
        rows.append({"angular_bin_pair":[a,b],"face_retention_ratio":fa/max(fd,1e-300),
                     "cell_retention_ratio":ca/max(cd,1e-300),"global_signed_retention_ratio":_retention(c),
                     "rowwise_signed_retention_ratio":_profile_retention(c,1),
                     "columnwise_signed_retention_ratio":_profile_retention(c,0),
                     "outer_x_quarters_absolute_share":outer/max(ca,1e-300),
                     "side_wall_eighths_absolute_share":side/max(ca,1e-300),
                     "face_to_cell_cancellation_ratio":ca/max(2*fa,1e-300),
                     "left_half_signed_sum_normalized":float(np.sum(c[:,:32])/max(ca,1e-300)),
                     "right_half_signed_sum_normalized":float(np.sum(c[:,32:])/max(ca,1e-300))})
    pf=np.stack(pf); pc=np.stack(pc); sf=np.sum(face,0); sc=np.sum(cell,0); tf=np.sum(pf,0); tc=np.sum(pc,0)
    closure={"maximum_absolute_error":max(float(np.max(np.abs(tf-sf))),float(np.max(np.abs(tc-sc)))),
             "face_relative_l2_error":_rel_l2(tf,sf),"cell_relative_l2_error":_rel_l2(tc,sc)}
    closure["within_guard"]=all(v<=CLOSURE_GUARD for k,v in closure.items() if k!="within_guard")
    minret=min(r["cell_retention_ratio"] for r in rows); maxrow=max(r["rowwise_signed_retention_ratio"] for r in rows)
    minouter=min(r["outer_x_quarters_absolute_share"] for r in rows); maxftc=max(r["face_to_cell_cancellation_ratio"] for r in rows)
    if minret<PAIR_RETENTION_GUARD: decision="stage83_unexpected_opposite_sector_cancellation_blocker"
    elif maxrow<=ROWWISE_SIGNED_RETENTION_GUARD and minouter>=OUTER_X_LOCALIZATION_GUARD and maxftc<=FACE_TO_CELL_CANCELLATION_GUARD:
        decision="stage83_rowwise_sidewall_localized_conservative_cancellation_stage84_wall_normal_sign_lobe_audit"
    elif maxrow<=ROWWISE_SIGNED_RETENTION_GUARD: decision="stage83_rowwise_spatial_cancellation_stage84_sign_lobe_geometry_audit"
    elif minouter>=OUTER_X_LOCALIZATION_GUARD: decision="stage83_sidewall_localized_without_rowwise_cancellation_stage84_sidewall_profile_audit"
    else: decision="stage83_diffuse_spatial_structure_no_solver_experiment_authorized"
    return {"pairs":rows,"pair_face_groups":pf,"pair_cell_groups":pc,"pair_reconstruction_closure":closure,
            "minimum_pair_cell_retention_ratio":minret,"maximum_pair_rowwise_signed_retention_ratio":maxrow,
            "minimum_pair_outer_x_quarters_share":minouter,"maximum_pair_face_to_cell_cancellation_ratio":maxftc,
            "decision":decision}


def build_summary(root):
    validate_stage83_design(); s82=_validate_stage82_artifact(root)
    with np.load(Path(root)/"within_sector_angular_coherence_maps.npz") as z:
        bins=np.array(z["vertical_oblique_bins"]); face=np.array(z["sector_face_groups"]); cell=np.array(z["sector_cell_divergence_groups"])
    m=spatial_cancellation_metrics(face,cell,bins); d=m["decision"]
    if d.endswith("wall_normal_sign_lobe_audit"):
        next_scope="Use only the checksum-verified Stage-83 pair maps to resolve fixed wall-normal x sign-lobe geometry and determine whether near-zero rowwise remainder is localized to a few side-wall sign changes or distributed across x; no solver rerun, retuning, validation claim, or cross-Knudsen extension is authorized."
    elif d.endswith("blocker"): next_scope="Stop and inspect Stage-82-to-83 pair construction; do not launch a solver experiment."
    else: next_scope="Continue only with the frozen spatial diagnostic named by the decision; no solver rerun or parameter change is authorized."
    cfg={"grid":[64,64],"kn0":10.0,"cold_hot_ratio":.1,"radial_nodes":40,"angular_nodes":96,"point_count":3840,
         "radial_scale":2.0,"limiter":"minmod","dominant_moment":DOMINANT_MOMENT,"dominant_radial_shell":2,
         "dominant_local_radial_node":1,"dominant_global_radial_node":21,"vertical_oblique_bins":[1,2,5,6],
         "opposite_sector_pairs":[[1,5],[2,6]],"outer_x_quarter_width":16,"side_wall_eighth_width":8,
         "pair_retention_guard":PAIR_RETENTION_GUARD,"rowwise_signed_retention_guard":ROWWISE_SIGNED_RETENTION_GUARD,
         "outer_x_localization_guard":OUTER_X_LOCALIZATION_GUARD,"face_to_cell_cancellation_guard":FACE_TO_CELL_CANCELLATION_GUARD,
         "solver_rerun_count":0,"physical_parameter_retuning":False,"collision_parameter_retuning":False,
         "correction_floor_retuning":False,"source_relaxation_retuning":False,"transport_parameter_retuning":False,
         "wall_model_retuning":False,"normalization_retuning":False,"velocity_quadrature_retuning":False,
         "failed_muscl_endpoint_rehabilitated":False,"cross_knudsen_extension_permitted":False,"validation_claim_permitted":False}
    summary={"stage":83,"description":"Frozen opposite-sector spatial cancellation and localization audit on Stage-82 node-21 maps.",
             "finite":bool(np.all(np.isfinite(m["pair_face_groups"])) and np.all(np.isfinite(m["pair_cell_groups"]))),
             "closure_closed":bool(m["pair_reconstruction_closure"]["within_guard"]),"configuration":cfg,
             "retained_stage82_decision":s82["decision"],"pair_reconstruction_closure":m["pair_reconstruction_closure"],
             "spatial_cancellation_metrics":{k:m[k] for k in ["pairs","minimum_pair_cell_retention_ratio","maximum_pair_rowwise_signed_retention_ratio","minimum_pair_outer_x_quarters_share","maximum_pair_face_to_cell_cancellation_ratio"]},
             "decision":d,"scientifically_justified_next_scope":next_scope,
             "positive_findings":["The two fixed opposite-sector pair maps reconstruct all four retained Stage-82 sector maps exactly without rebucketing or a new solver state.","Global, rowwise, columnwise, side-wall, and face-to-cell cancellation are quantified on the frozen pair maps."],
             "negative_findings":["This is a frozen residual-structure diagnostic, not an adjoint sensitivity and not evidence that changing velocity ordinates, transport, or wall treatment would improve q_av.","Strong rowwise or side-wall structure does not constitute physical validation or by itself identify a numerical error mechanism.","The failed Stage-28 MUSCL endpoint remains unrecovered and is not extended across Knudsen number.","No physical, collision, correction-floor, source-relaxation, transport, wall, normalization, or velocity-quadrature parameter is retuned, and no cavity solver is rerun."]}
    maps={"opposite_sector_pairs":np.array(OPPOSITE_SECTOR_PAIRS,np.int16),"pair_face_groups":m["pair_face_groups"],
          "pair_cell_divergence_groups":m["pair_cell_groups"],"vertical_oblique_bins":bins,
          "retained_stage82_sector_face_groups":face,"retained_stage82_sector_cell_divergence_groups":cell}
    return summary,maps


def run(root,out):
    summary,maps=build_summary(root); out=Path(out); out.mkdir(parents=True,exist_ok=True)
    (out/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    np.savez_compressed(out/"opposite_sector_spatial_cancellation_localization_maps.npz",**maps); return summary


def main():
    p=argparse.ArgumentParser(); p.add_argument("--stage82-artifact-dir",required=True); p.add_argument("--output-dir",required=True); a=p.parse_args()
    print(json.dumps(run(a.stage82_artifact_dir,a.output_dir),indent=2,sort_keys=True))

if __name__=="__main__": main()
