import numpy as np
from vgdsmc.simulator import CavityConfig, run_cavity
from vgdsmc.dataset import generate_case


def test_simulator_and_dataset(tmp_path):
    cfg = CavityConfig(nx=6, ny=6, particles_per_cell=3, steps=10, sample_start=4, seed=2)
    fields = run_cavity(cfg)
    assert fields["T"].shape == (6, 6)
    assert np.isfinite(fields["T"]).all()
    path = generate_case(tmp_path, cfg, reference_ppc=6)
    data = np.load(path)
    assert data["x"].shape == (4, 6, 6)
    assert data["label"].shape == (6, 6)
