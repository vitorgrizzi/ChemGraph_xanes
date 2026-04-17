import json
from pathlib import Path

from ase import Atoms
from ase.db import connect

from chemgraph.tools.xanes_tools import prepare_xanes_batch


def _build_test_db(db_path: Path) -> None:
    db = connect(str(db_path))
    db.write(Atoms("CuO", positions=[[0, 0, 0], [1.8, 0, 0]]), name="cuo_1")
    db.write(Atoms("Cu2O", positions=[[0, 0, 0], [1.8, 0, 0], [0, 1.8, 0]]), name="cu2o_1")


def test_prepare_xanes_batch_from_ase_db(tmp_path):
    db_path = tmp_path / "structures.db"
    output_dir = tmp_path / "xanes_output"
    _build_test_db(db_path)

    batch = prepare_xanes_batch(
        input_source=str(db_path),
        output_dir=str(output_dir),
        z_absorber=29,
    )

    runs_dir = Path(batch["runs_dir"])
    assert batch["n_total"] == 2
    assert batch["n_prepared"] == 2
    assert batch["n_skipped"] == 0
    assert runs_dir.exists()

    for idx in range(2):
        run_dir = runs_dir / f"run_{idx}"
        assert (run_dir / "fdmfile.txt").exists()
        assert (run_dir / "fdmnes_in.txt").exists()
        assert (run_dir / "run_metadata.json").exists()
        assert list(run_dir.glob("*.pkl"))

        with open(run_dir / "run_metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)
        assert metadata["z_absorber"] == 29
        assert metadata["run_dir"] == str(run_dir)
        assert "::id=" in metadata["source"]


def test_prepare_xanes_batch_skips_completed_runs(tmp_path):
    db_path = tmp_path / "structures.db"
    output_dir = tmp_path / "xanes_output"
    _build_test_db(db_path)

    first_batch = prepare_xanes_batch(
        input_source=str(db_path),
        output_dir=str(output_dir),
        z_absorber=29,
    )

    run_0 = Path(first_batch["runs_dir"]) / "run_0"
    with open(run_0 / "existing_conv.txt", "w", encoding="utf-8") as f:
        f.write("header\n")
        f.write("0 1\n" * 600)

    second_batch = prepare_xanes_batch(
        input_source=str(db_path),
        output_dir=str(output_dir),
        z_absorber=29,
        skip_completed=True,
    )

    assert second_batch["n_total"] == 2
    assert second_batch["n_prepared"] == 1
    assert second_batch["n_skipped"] == 1
    assert second_batch["jobs"][0]["status"] == "skipped_existing"
    assert second_batch["jobs"][1]["status"] == "prepared"
