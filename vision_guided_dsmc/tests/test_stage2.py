import numpy as np

from vgdsmc.adaptive import allocation_summary, label_to_target_ppc, reallocate_particles


def test_allocation_map():
    label = np.array([[0, 1], [2, 1]])
    target = label_to_target_ppc(label, base_ppc=20)
    assert target.tolist() == [[10, 20], [60, 20]]
    assert allocation_summary(label, base_ppc=20)["adaptive_particles"] == 110


def test_reallocation_exact_counts():
    rng = np.random.default_rng(1)
    pos = rng.random((20, 2))
    vel = rng.normal(size=(20, 2))
    target = np.array([[3, 4], [5, 6]])
    new_pos, new_vel = reallocate_particles(pos, vel, target, rng)
    assert len(new_pos) == 18
    assert new_pos.shape == new_vel.shape
    assert np.all((new_pos >= 0.0) & (new_pos < 1.0))
