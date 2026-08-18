import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HPC = ROOT / "hpc"


class MaxwellCampaignAssetTests(unittest.TestCase):
    scripts = (
        HPC / "unity_sparta_maxwell_kngu005_020_jfm_build.slurm",
        HPC / "unity_sparta_maxwell_kngu005_020_jfm_single.slurm",
        HPC / "unity_sparta_maxwell_kngu005_020_jfm_collect.slurm",
        HPC / "bootstrap_unity_sparta_maxwell_kngu005_020_jfm.sh",
    )

    def test_all_shell_assets_parse(self) -> None:
        for script in self.scripts:
            with self.subTest(script=script.name):
                self.assertTrue(script.is_file())
                subprocess.run(["bash", "-n", str(script)], check=True)

    def test_single_realisation_contract(self) -> None:
        text = self.scripts[1].read_text(encoding="utf-8")
        for required in (
            "#SBATCH --array=0-1%2",
            "#SBATCH --ntasks=16",
            "#SBATCH --mem=64G",
            "KN_VALUES=(0.05 0.20)",
            "MAXWELL_SINGLE_SEED:-104729",
            "grid.final.00200000",
            "f_fieldavg[15]",
            "--require-final",
        ):
            self.assertIn(required, text)

    def test_build_is_pinned_and_smokes_extended_schema(self) -> None:
        text = self.scripts[0].read_text(encoding="utf-8")
        self.assertIn("912c9e163c38ea5c3562d039e65215f6e2a4f3f8", text)
        self.assertIn("'005 0.05' '020 0.20'", text)
        self.assertIn("f_fieldavg[15]", text)
        self.assertIn("sonine/grid", text)
        self.assertIn("pflux/grid", text)

    def test_collector_makes_a_small_integrity_checked_zip(self) -> None:
        text = self.scripts[2].read_text(encoding="utf-8")
        self.assertIn("_TO_ANALYZE.zip", text)
        self.assertIn("FILES.sha256", text)
        self.assertIn("archive.testzip()", text)
        self.assertIn("grid.final.00200000", text)
        self.assertNotIn("grid.checkpoint.*", text)
        self.assertNotIn("restart.maxwell", text)

    def test_bootstrap_targets_published_branch_name(self) -> None:
        text = self.scripts[3].read_text(encoding="utf-8")
        self.assertIn("agent/maxwell-matched-antifourier", text)
        self.assertIn("afterok:${BUILD_JOB_ID}", text)
        self.assertIn("afterany:${RUN_JOB_ID}", text)


if __name__ == "__main__":
    unittest.main()
