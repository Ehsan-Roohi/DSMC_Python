#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_jcp3_ds2v_m12.py"
SPEC = importlib.util.spec_from_file_location("prepare_jcp3", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_data_patch_changes_exactly_one_speed_token() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "DS2VD.DAT"
        output = root / "patched.DAT"
        source.write_text("1\n4.247E20\n200.0\n2634.1\n500.0\n", encoding="utf-8")
        report = MODULE.patch_data(source, output)
        assert report["changed_token_count"] == 1
        assert "3.16092000E+03" in output.read_text(encoding="utf-8")
        assert "2634.1" not in output.read_text(encoding="utf-8")


def test_data_patch_rejects_ambiguous_speed_tokens() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "DS2VD.DAT"
        output = root / "patched.DAT"
        source.write_text("2634.1\n2.6341E3\n", encoding="utf-8")
        try:
            MODULE.patch_data(source, output)
        except ValueError as error:
            assert "exactly one" in str(error)
        else:
            raise AssertionError("ambiguous Mach token was accepted")


def test_replace_once_is_fail_closed() -> None:
    assert MODULE.replace_once("abc", "b", "B", "x") == "aBc"
    for text in ("ac", "bb"):
        try:
            MODULE.replace_once(text, "b", "B", "x")
        except ValueError:
            pass
        else:
            raise AssertionError("non-unique anchor was accepted")


if __name__ == "__main__":
    test_data_patch_changes_exactly_one_speed_token()
    test_data_patch_rejects_ambiguous_speed_tokens()
    test_replace_once_is_fail_closed()
    print("3 JCP3 preparation tests passed")
