from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path
import numpy as np

STAGE83_COMPLETED_ENDPOINT = {
    "workflow_run_id": 31225050170,
    "workflow_job_id": 93017544665,
    "workflow_conclusion": "success",
    "tests_passed": 268,
    "tests_failed": 0,
    "artifact_id": 9014969851,
    "artifact_sha256": "31f28e8e1e671ab7cf83255f6813a2d5472ef7139c685ae42c6de8b9414f2874",
    "source_head_sha": "d13a0f2f21d3f791875428d04f2b74c140952947",
    "summary_sha256": "4a071fdf1acd320ad87ec2e3d05761364aadab914defdbd89dd69e2fa61bbb2e",
    "maps_sha256": "6f6b83f5b20e732ff5867ed57aeeb99f25d40060a89e786e0eefea333609bf8c",
    "decision": "stage83_rowwise_sidewall_localized_conservative_cancellation_stage84_wall_normal_sign_lobe_audit",
}
GRID=(64,64); KNUDSEN=10.0; COLD_HOT_RATIO=0.1; RADIAL_NODES=40; ANGULAR_NODES=96
POINT_COUNT=3840; RADIAL_SCALE=2.0; LIMITER="minmod"; DOMINANT_MOMENT="transverse_kinetic"
DOMINANT_RADIAL_SHELL=2; DOMINANT_LOCAL_RADIAL_NODE=1; DOMINANT_GLOBAL_RADIAL_NODE=21
VERTICAL_OBLIQUE_BINS=(1,2,5,6); OPPOSITE_SECTOR_PAIRS=((1,5),(2,6))
OUTERMOST_CELL_SHARE_MAX=.05; FIRST_INTERIOR_TOTAL_SHARE_GUARD=.30
NEGATIVE_FIRST_INTERIOR_SHARE_GUARD=.75; POSITIVE_FIRST_INTERIOR_SHARE_MAX=.10
SIDEWALL_TWO_CELL_SHARE_GUARD=.50; CLOSURE_GUARD=1e-10


def validate_stage84_design(**overrides):
    frozen={"grid":GRID,"kn0":KNUDSEN,"cold_hot_ratio":COLD_HOT_RATIO,"radial_nodes":RADIAL_NODES,
            "angular_nodes":ANGULAR_NODES,"radial_scale":RADIAL_SCALE,"limiter":LIMITER,
            "vertical_oblique_bins":VERTICAL_OBLIQUE_BINS,"opposite_sector_pairs":OPPOSITE_SECTOR_PAIRS}
    for k,v in overrides.items():
        if k not in frozen or v != frozen[k]:
            raise ValueError("Stage 84 is frozen to the completed Stage-83 endpoint; no solver or parameter retuning is permitted")


def _sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest()


def _validate_stage83_artifact(root):
    root=Path(root); summary=root/"summary.json"; maps=root/"opposite_sector_spatial_cancellation_localization_maps.npz"
    if not summary.exists() or not maps.exists(): raise FileNotFoundError("Stage 84 requires exact Stage-83 summary and maps")
    if _sha256(summary)!=STAGE83_COMPLETED_ENDPOINT["summary_sha256"]: raise ValueError("Stage-83 summary checksum mismatch")
    if _sha256(maps)!=STAGE83_COMPLETED_ENDPOINT["maps_sha256"]: raise ValueError("Stage-83 maps checksum mismatch")
    s=json.loads(summary.read_text())
    if s.get("stage")!=83 or s.get("decision")!=STAGE83_COMPLETED_ENDPOINT["decision"]: raise ValueError("Stage-83 decision mismatch")
    if not s.get("finite") or not s.get("closure_closed"): raise ValueError("Stage-83 endpoint is not finite and closure-closed")
    m=s["spatial_cancellation_metrics"]
    if m["minimum_pair_cell_retention_ratio"]<.75 or m["maximum_pair_rowwise_signed_retention_ratio"]>.10:
        raise ValueError("Stage-83 decision branch does not authorize Stage 84")
    c=s["configuration"]
    expected={"grid":[64,64],"kn0":10.0,"radial_nodes":40,"angular_nodes":96,"point_count":3840,
              "radial_scale":2.0,"limiter":"minmod","dominant_moment":DOMINANT_MOMENT,
              "dominant_radial_shell":2,"dominant_local_radial_node":1,"dominant_global_radial_node":21,
              "vertical_oblique_bins":[1,2,5,6],"opposite_sector_pairs":[[1,5],[2,6]],"solver_rerun_count":0,
              "failed_muscl_endpoint_rehabilitated":False,"cross_knudsen_extension_permitted":False,
              "validation_claim_permitted":False}
    for k,v in expected.items():
        if c.get(k)!=v: raise ValueError(f"Stage-83 frozen configuration mismatch for {k}")
    if any(v is not False for k,v in c.items() if k.endswith("_retuning")): raise ValueError("Stage-83 contains retuning")
    return s


def conservative_cell_from_face(face):
    face=np.asarray(face,float)
    if face.shape[-2:]!=(64,63): raise ValueError("Stage 84 requires 64x63 x-face maps")
    cell=np.zeros(face.shape[:-1]+(64,),dtype=np.float64)
    cell[...,0]=-face[...,0]
    cell[...,1:-1]=face[...,:-1]-face[...,1:]
    cell[...,-1]=face[...,-1]
    return cell


def _rel_l2(a,b):
    d=float(np.linalg.norm(np.ravel(b))); return 0.0 if d<=1e-300 else float(np.linalg.norm(np.ravel(a-b))/d)


def _effective_support(a):
    a=np.asarray(a,float); m=float(np.sum(a)); q=float(np.sum(a*a))
    return 0.0 if m<=1e-300 or q<=1e-300 else float(m*m/q)


def _weighted_sign_support(cell,positive=True):
    cell=np.asarray(cell,float); row_l1=np.sum(np.abs(cell),axis=1); total=float(np.sum(row_l1))
    values=[]
    for row in cell:
        a=np.where(row>0,row,0.0) if positive else np.where(row<0,-row,0.0)
        values.append(_effective_support(a))
    if total<=1e-300: return 0.0
    return float(np.sum(np.asarray(values)*row_l1)/total)


def sign_lobe_geometry_metrics(face,cell,pairs):
    face=np.asarray(face,float); cell=np.asarray(cell,float); pairs=np.asarray(pairs,np.int16)
    if face.shape!=(2,64,63) or cell.shape!=(2,64,64): raise ValueError("Stage 84 requires exact Stage-83 pair-map shapes")
    if not np.array_equal(pairs,[[1,5],[2,6]]): raise ValueError("Stage 84 requires exact inherited opposite-sector pairs")
    if not np.all(np.isfinite(face)) or not np.all(np.isfinite(cell)): raise ValueError("Stage 84 requires finite maps")
    reconstructed=conservative_cell_from_face(face)
    closure={"maximum_absolute_error":float(np.max(np.abs(reconstructed-cell))),
             "cell_relative_l2_error":_rel_l2(reconstructed,cell)}
    closure["within_guard"]=all(v<=CLOSURE_GUARD for k,v in closure.items() if k!="within_guard")
    rows=[]
    for i,pair in enumerate(pairs.tolist()):
        c=cell[i]; total=float(np.sum(np.abs(c))); pos=np.where(c>0,c,0.0); neg=np.where(c<0,-c,0.0)
        pos_mass=float(np.sum(pos)); neg_mass=float(np.sum(neg))
        outer=float(np.sum(np.abs(c[:,0]))+np.sum(np.abs(c[:,-1])))
        first=float(np.sum(np.abs(c[:,1]))+np.sum(np.abs(c[:,-2])))
        side2=outer+first
        neg_first=float(np.sum(neg[:,1])+np.sum(neg[:,-2]))
        pos_first=float(np.sum(pos[:,1])+np.sum(pos[:,-2]))
        row_signed=float(np.sum(np.abs(np.sum(c,axis=1))))/max(total,1e-300)
        rows.append({"angular_bin_pair":pair,
                     "rowwise_signed_retention_ratio":row_signed,
                     "positive_mass_fraction":pos_mass/max(total,1e-300),
                     "negative_mass_fraction":neg_mass/max(total,1e-300),
                     "outermost_cells_absolute_share":outer/max(total,1e-300),
                     "first_interior_cells_absolute_share":first/max(total,1e-300),
                     "sidewall_two_cell_absolute_share":side2/max(total,1e-300),
                     "negative_mass_first_interior_share":neg_first/max(neg_mass,1e-300),
                     "positive_mass_first_interior_share":pos_first/max(pos_mass,1e-300),
                     "negative_effective_support_width_cells":_weighted_sign_support(c,positive=False),
                     "positive_effective_support_width_cells":_weighted_sign_support(c,positive=True)})
    min_first=min(r["first_interior_cells_absolute_share"] for r in rows)
    min_neg_first=min(r["negative_mass_first_interior_share"] for r in rows)
    max_pos_first=max(r["positive_mass_first_interior_share"] for r in rows)
    max_outer=max(r["outermost_cells_absolute_share"] for r in rows)
    min_side2=min(r["sidewall_two_cell_absolute_share"] for r in rows)
    max_row=max(r["rowwise_signed_retention_ratio"] for r in rows)
    max_neg_width=max(r["negative_effective_support_width_cells"] for r in rows)
    min_pos_width=min(r["positive_effective_support_width_cells"] for r in rows)
    if not closure["within_guard"]:
        decision="stage84_face_to_cell_divergence_closure_blocker"
    elif min_first>=FIRST_INTERIOR_TOTAL_SHARE_GUARD and min_neg_first>=NEGATIVE_FIRST_INTERIOR_SHARE_GUARD and max_pos_first<=POSITIVE_FIRST_INTERIOR_SHARE_MAX and max_outer<=OUTERMOST_CELL_SHARE_MAX:
        decision="stage84_first_interior_negative_lobes_broad_compensation_stage85_near_wall_face_amplitude_suppression_audit"
    elif min_side2>=SIDEWALL_TWO_CELL_SHARE_GUARD:
        decision="stage84_wall_band_sign_lobes_stage85_wall_band_face_profile_audit"
    else:
        decision="stage84_distributed_sign_lobes_no_solver_experiment_authorized"
    return {"pairs":rows,"face_to_cell_divergence_closure":closure,
            "minimum_first_interior_cells_absolute_share":min_first,
            "minimum_negative_mass_first_interior_share":min_neg_first,
            "maximum_positive_mass_first_interior_share":max_pos_first,
            "maximum_outermost_cells_absolute_share":max_outer,
            "minimum_sidewall_two_cell_absolute_share":min_side2,
            "maximum_rowwise_signed_retention_ratio":max_row,
            "maximum_negative_effective_support_width_cells":max_neg_width,
            "minimum_positive_effective_support_width_cells":min_pos_width,
            "decision":decision}


def build_summary(root):
    validate_stage84_design(); s83=_validate_stage83_artifact(root)
    with np.load(Path(root)/"opposite_sector_spatial_cancellation_localization_maps.npz") as z:
        pairs=np.array(z["opposite_sector_pairs"]); face=np.array(z["pair_face_groups"]); cell=np.array(z["pair_cell_divergence_groups"])
    m=sign_lobe_geometry_metrics(face,cell,pairs); d=m["decision"]
    if d.endswith("near_wall_face_amplitude_suppression_audit"):
        next_scope="Use only the checksum-verified Stage-84 retained pair-face maps to compare the wall-adjacent and first-interior x-face amplitudes and their y coherence, determining whether the compact first-interior cell lobes arise from abrupt near-wall face-amplitude suppression rather than distributed interior variation; no solver rerun, retuning, validation claim, or cross-Knudsen extension is authorized."
    elif d.endswith("blocker"):
        next_scope="Stop and inspect Stage-83 face-to-cell construction; do not launch a solver experiment."
    else:
        next_scope="Continue only with the frozen face-profile diagnostic named by the decision; no solver rerun or parameter change is authorized."
    cfg={"grid":[64,64],"kn0":10.0,"cold_hot_ratio":.1,"radial_nodes":40,"angular_nodes":96,"point_count":3840,
         "radial_scale":2.0,"limiter":"minmod","dominant_moment":DOMINANT_MOMENT,"dominant_radial_shell":2,
         "dominant_local_radial_node":1,"dominant_global_radial_node":21,"vertical_oblique_bins":[1,2,5,6],
         "opposite_sector_pairs":[[1,5],[2,6]],"outermost_cell_share_max":OUTERMOST_CELL_SHARE_MAX,
         "first_interior_total_share_guard":FIRST_INTERIOR_TOTAL_SHARE_GUARD,
         "negative_first_interior_share_guard":NEGATIVE_FIRST_INTERIOR_SHARE_GUARD,
         "positive_first_interior_share_max":POSITIVE_FIRST_INTERIOR_SHARE_MAX,
         "sidewall_two_cell_share_guard":SIDEWALL_TWO_CELL_SHARE_GUARD,
         "solver_rerun_count":0,"physical_parameter_retuning":False,"collision_parameter_retuning":False,
         "correction_floor_retuning":False,"source_relaxation_retuning":False,"transport_parameter_retuning":False,
         "wall_model_retuning":False,"normalization_retuning":False,"velocity_quadrature_retuning":False,
         "failed_muscl_endpoint_rehabilitated":False,"cross_knudsen_extension_permitted":False,"validation_claim_permitted":False}
    summary={"stage":84,"description":"Frozen wall-normal sign-lobe geometry audit on Stage-83 opposite-sector pair maps.",
             "finite":bool(np.all(np.isfinite(face)) and np.all(np.isfinite(cell))),
             "closure_closed":bool(m["face_to_cell_divergence_closure"]["within_guard"]),"configuration":cfg,
             "retained_stage83_decision":s83["decision"],"face_to_cell_divergence_closure":m["face_to_cell_divergence_closure"],
             "sign_lobe_geometry_metrics":{k:m[k] for k in ["pairs","minimum_first_interior_cells_absolute_share",
                 "minimum_negative_mass_first_interior_share","maximum_positive_mass_first_interior_share",
                 "maximum_outermost_cells_absolute_share","minimum_sidewall_two_cell_absolute_share",
                 "maximum_rowwise_signed_retention_ratio","maximum_negative_effective_support_width_cells",
                 "minimum_positive_effective_support_width_cells"]},
             "decision":d,"scientifically_justified_next_scope":next_scope,
             "positive_findings":["The Stage-83 pair cell maps are reconstructed directly from the retained x-face maps by the conservative discrete divergence to numerical precision.","Wall-normal positive and negative signed masses, first-interior versus outermost-cell localization, and sign-specific effective support widths are quantified without a new solver state."],
             "negative_findings":["The near-zero rowwise signed sum is a structural telescoping property of the conservative internal-face divergence and is not an independent physical balance.","A compact near-wall sign lobe does not establish a wall-model or reconstruction error and is not an adjoint sensitivity or evidence that changing transport would improve q_av.","The failed Stage-28 MUSCL endpoint remains unrecovered and is not extended across Knudsen number.","No physical, collision, correction-floor, source-relaxation, transport, wall, normalization, or velocity-quadrature parameter is retuned, and no cavity solver is rerun."]}
    maps={"opposite_sector_pairs":pairs,"pair_face_groups":face,"pair_cell_divergence_groups":cell,
          "conservative_reconstructed_cell_divergence_groups":conservative_cell_from_face(face)}
    return summary,maps


def run(root,out):
    summary,maps=build_summary(root); out=Path(out); out.mkdir(parents=True,exist_ok=True)
    (out/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    np.savez_compressed(out/"wall_normal_sign_lobe_geometry_maps.npz",**maps); return summary


def main():
    p=argparse.ArgumentParser(); p.add_argument("--stage83-artifact-dir",required=True); p.add_argument("--output-dir",required=True); a=p.parse_args()
    print(json.dumps(run(a.stage83_artifact_dir,a.output_dir),indent=2,sort_keys=True))

if __name__=="__main__": main()
