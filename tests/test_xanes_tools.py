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


def test_prepare_xanes_batch_writes_energy_range(tmp_path):
    db_path = tmp_path / "structures.db"
    output_dir = tmp_path / "xanes_output"
    _build_test_db(db_path)

    batch = prepare_xanes_batch(
        input_source=str(db_path),
        output_dir=str(output_dir),
        z_absorber=29,
        energy_range=[-5.0, 0.5, 60.0],
    )

    run_dir = Path(batch["runs_dir"]) / "run_0"
    fdmnes_input = (run_dir / "fdmnes_in.txt").read_text(encoding="utf-8")
    assert "Range\n-5 0.5 60\n\n" in fdmnes_input

    with open(run_dir / "run_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    assert metadata["energy_range"] == [-5.0, 0.5, 60.0]


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


def test_get_normalized_xanes_normal(tmp_path):
    import numpy as np
    from chemgraph.tools.xanes_tools import get_normalized_xanes

    conv_file = tmp_path / "test_conv.txt"
    lines = [
        "Energy Mu",
        "-30.0 0.1",
        "-25.0 0.1",
        "-20.0 0.1",
        "0.0 0.5",
        "10.0 0.8",
        "20.0 0.9",
        "50.0 1.1",
        "60.0 1.2",
        "70.0 1.3"
    ]
    conv_file.write_text("\n".join(lines), encoding="utf-8")

    norm, raw = get_normalized_xanes(
        conv_file,
        pre_edge_width=20.0,
        post_edge_width=50.0,
    )
    assert norm.shape == (9, 2)
    assert np.allclose(raw[:, 0], norm[:, 0])


def test_get_normalized_xanes_fallback_pre_edge(tmp_path):
    import numpy as np
    from chemgraph.tools.xanes_tools import get_normalized_xanes

    conv_file = tmp_path / "test_conv.txt"
    lines = [
        "Energy Mu",
        "-30.0 0.1",
        "-25.0 0.1",
        "-20.0 0.1",
        "0.0 0.5",
        "10.0 0.8",
        "20.0 0.9",
        "50.0 1.1",
        "60.0 1.2",
        "70.0 1.3"
    ]
    conv_file.write_text("\n".join(lines), encoding="utf-8")

    norm, raw = get_normalized_xanes(
        conv_file,
        pre_edge_width=40.0,
        post_edge_width=50.0,
    )
    expected_norm_mu = raw[:, 1] / 1.3
    assert np.allclose(norm[:, 1], expected_norm_mu)


def test_get_normalized_xanes_fallback_post_edge(tmp_path):
    import numpy as np
    from chemgraph.tools.xanes_tools import get_normalized_xanes

    conv_file = tmp_path / "test_conv.txt"
    lines = [
        "Energy Mu",
        "-30.0 0.1",
        "-25.0 0.1",
        "-20.0 0.1",
        "0.0 0.5",
        "10.0 0.8",
        "20.0 0.9",
        "50.0 1.1",
        "60.0 1.2",
        "70.0 1.3"
    ]
    conv_file.write_text("\n".join(lines), encoding="utf-8")

    norm, raw = get_normalized_xanes(
        conv_file,
        pre_edge_width=20.0,
        post_edge_width=65.0,
    )
    expected_norm_mu = raw[:, 1] / 1.3
    assert np.allclose(norm[:, 1], expected_norm_mu)


def test_prepare_xanes_batch_writes_absorber_idx(tmp_path):
    db_path = tmp_path / "structures.db"
    output_dir = tmp_path / "xanes_output"
    _build_test_db(db_path)

    batch = prepare_xanes_batch(
        input_source=str(db_path),
        output_dir=str(output_dir),
        z_absorber=29,
        absorber_idx=2,
    )

    run_dir = Path(batch["runs_dir"]) / "run_0"
    fdmnes_input = (run_dir / "fdmnes_in.txt").read_text(encoding="utf-8")
    assert "Absorber\n2\n\n" in fdmnes_input
    assert "Z_absorber" not in fdmnes_input

    with open(run_dir / "run_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    assert metadata["absorber_idx"] == 2


def test_prepare_xanes_batch_writes_edge(tmp_path):
    db_path = tmp_path / "structures.db"
    output_dir = tmp_path / "xanes_output"
    _build_test_db(db_path)

    batch = prepare_xanes_batch(
        input_source=str(db_path),
        output_dir=str(output_dir),
        z_absorber=29,
        edge="L3",
    )

    run_dir = Path(batch["runs_dir"]) / "run_0"
    fdmnes_input = (run_dir / "fdmnes_in.txt").read_text(encoding="utf-8")
    assert "Edge\nL3\n\n" in fdmnes_input

    with open(run_dir / "run_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    assert metadata["edge"] == "L3"


def test_prepare_xanes_batch_writes_custom_boolean_flags(tmp_path):
    db_path = tmp_path / "structures.db"
    output_dir = tmp_path / "xanes_output"
    _build_test_db(db_path)

    # test custom flags set to False
    batch = prepare_xanes_batch(
        input_source=str(db_path),
        output_dir=str(output_dir),
        z_absorber=29,
        green=False,
        density_all=False,
        quadrupole=False,
        spherical=False,
        scf=False,
    )

    run_dir = Path(batch["runs_dir"]) / "run_0"
    fdmnes_input = (run_dir / "fdmnes_in.txt").read_text(encoding="utf-8")
    assert "Green\n" not in fdmnes_input
    assert "Density_all\n" not in fdmnes_input
    assert "Quadrupole\n" not in fdmnes_input
    assert "Spherical\n" not in fdmnes_input
    assert "SCF\n" not in fdmnes_input

    with open(run_dir / "run_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    assert metadata["green"] is False
    assert metadata["density_all"] is False
    assert metadata["quadrupole"] is False
    assert metadata["spherical"] is False
    assert metadata["scf"] is False


def test_get_normalized_xanes_E0_in_domain(tmp_path):
    import numpy as np
    from chemgraph.tools.xanes_tools import get_normalized_xanes

    conv_file = tmp_path / "test_conv.txt"
    lines = [
        "Energy Mu",
        "-30.0 0.1",
        "-20.0 0.1",
        "-10.0 0.1",
        "0.0 0.5",
        "10.0 0.8",
        "20.0 0.9",
        "30.0 1.1",
        "40.0 1.2",
        "50.0 1.3"
    ]
    conv_file.write_text("\n".join(lines), encoding="utf-8")

    norm, raw = get_normalized_xanes(
        conv_file,
        pre_edge_width=20.0,
        post_edge_width=30.0,
    )
    assert norm.shape == (9, 2)


def test_get_normalized_xanes_E0_not_in_domain(tmp_path):
    import numpy as np
    from chemgraph.tools.xanes_tools import get_normalized_xanes

    conv_file = tmp_path / "test_conv.txt"
    lines = [
        "Energy Mu",
        "8000.0 0.1",
        "8005.0 0.1",
        "8010.0 0.5",
        "8015.0 0.8",
        "8020.0 0.9",
        "8025.0 1.1",
        "8030.0 1.2",
        "8040.0 1.2",
        "8050.0 1.3"
    ]
    conv_file.write_text("\n".join(lines), encoding="utf-8")

    norm, raw = get_normalized_xanes(
        conv_file,
        pre_edge_width=5.0,
        post_edge_width=15.0,
    )
    assert norm.shape == (9, 2)
    assert np.isclose(norm[0, 1], 0.0, atol=1e-2)





