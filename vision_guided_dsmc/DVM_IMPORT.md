# Import the existing DVM cavity reference

The existing cluster configuration points to:

```text
data/cavity_dvm/cavity_dvm_moments.dat
```

After installing this package, run:

```bash
cd vision_guided_dsmc
python -m pip install -e '.[full]'

DVM_MOMENTS=data/cavity_dvm/cavity_dvm_moments.dat \
DVM_REFERENCE_OUT=outputs/dvm/DVM65_reference.npz \
bash scripts/import_cluster_dvm_reference.sh
```

The importer accepts Tecplot `VARIABLES=...` files, ordinary named-column tables, and headerless whitespace tables. For a headerless file, provide the full order explicitly:

```bash
DVM_COLUMNS='x,y,rho,u,v,T,qx,qy,thetax,thetay,thetaz,sigxx,sigyy,sigxy,m3x,m3y,sx,sy,m4x,m4y,kx,ky' \
bash scripts/import_cluster_dvm_reference.sh
```

The standardized output contains at least:

```text
x, y, rho, u, v, T
```

and is accepted by the deterministic-reference supervision pipeline:

```bash
vgdsmc-reference-case \
  --coarse-case outputs/coarse/case.npz \
  --reference outputs/dvm/DVM65_reference.npz \
  --output outputs/supervised/DVM65_case.npz
```

The importer verifies a complete tensor-product grid, finite values, and positive density and temperature. A sidecar JSON records the case label and operating parameters.
