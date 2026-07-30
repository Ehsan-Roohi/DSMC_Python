from pathlib import Path

import numpy as np
import pytest

from vgdsmc.dvm_import import import_dvm_table, write_reference_npz


def test_import_tecplot_table_and_write_npz(tmp_path: Path):
    source = tmp_path / "dvm.dat"
    source.write_text(
        '\n'.join([
            'TITLE="DVM cavity"',
            'VARIABLES="X","Y","RHO","U","V","T"',
            'ZONE I=2, J=2, F=POINT',
            '1 1 1.03 0.3 0.4 1.3',
            '0 0 1.00 0.0 0.1 1.0',
            '1 0 1.01 0.1 0.2 1.1',
            '0 1 1.02 0.2 0.3 1.2',
        ]),
        encoding="utf-8",
    )
    grid = import_dvm_table(source)
    assert grid.fields["T"].shape == (2, 2)
    assert np.isclose(grid.fields["T"][1, 1], 1.3)
    output = write_reference_npz(grid, tmp_path / "reference.npz", {"case": "DVM65"})
    with np.load(output) as data:
        assert set(("x", "y", "rho", "u", "v", "T")).issubset(data.files)
        assert np.all(data["rho"] > 0.0)


def test_import_headerless_with_explicit_columns(tmp_path: Path):
    source = tmp_path / "moments.dat"
    source.write_text(
        "0 0 1 0 0 1 99\n1 0 1.1 0.1 0 1.2 98\n0 1 1.2 0 0.1 1.3 97\n1 1 1.3 0.1 0.1 1.4 96\n",
        encoding="utf-8",
    )
    grid = import_dvm_table(source, "x,y,rho,u,v,T,qx")
    assert grid.fields["u"].shape == (2, 2)
    assert np.isclose(grid.fields["v"][1, 1], 0.1)


def test_import_rejects_incomplete_grid(tmp_path: Path):
    source = tmp_path / "bad.dat"
    source.write_text(
        "x y rho u v T\n0 0 1 0 0 1\n1 0 1 0 0 1\n0 1 1 0 0 1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="complete tensor grid"):
        import_dvm_table(source)


def test_import_rejects_nonpositive_temperature(tmp_path: Path):
    source = tmp_path / "bad_temperature.dat"
    source.write_text(
        "x y rho u v T\n0 0 1 0 0 1\n1 0 1 0 0 1\n0 1 1 0 0 1\n1 1 1 0 0 0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="positive"):
        import_dvm_table(source)
