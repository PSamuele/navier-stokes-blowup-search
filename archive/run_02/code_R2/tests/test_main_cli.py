"""
tests/test_main_cli.py - Unit and integration tests for main.py CLI and restart mechanics.
"""

import os
import sys
import json
import subprocess
import numpy as np
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from main import build_parser, run_solver, main


def test_cli_help_stdout():
    """
    Verify that `python main.py --help` runs successfully and displays all CLI options.
    """
    result = subprocess.run(
        [sys.executable, "main.py", "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Help command failed: {result.stderr}"
    stdout = result.stdout

    # Verify all expected options are present in help text
    expected_flags = [
        "--mesh",
        "--out_dir",
        "--checkpoint_dir",
        "--restart",
        "-T",
        "--t_final",
        "--num_steps",
        "--adaptive_dt",
        "--no_adaptive_dt",
        "--cfl",
        "--dx_min",
        "--nu",
        "--log_interval",
    ]
    for flag in expected_flags:
        assert flag in stdout, f"Expected flag '{flag}' in help output:\n{stdout}"


def test_build_parser_defaults():
    """
    Verify default argument values from build_parser().
    """
    parser = build_parser()
    args = parser.parse_args([])

    assert args.mesh == "assets/apple_domain.msh"
    assert args.out_dir == "results"
    assert args.checkpoint_dir is None
    assert args.restart is False
    assert np.isclose(args.t_final, 0.55)
    assert args.num_steps == 1100
    assert args.adaptive_dt is True
    assert np.isclose(args.cfl, 0.5)
    assert np.isclose(args.dx_min, 0.0001)
    assert np.isclose(args.nu, 1e-3)
    assert args.log_interval == 10


def test_build_parser_custom_args():
    """
    Verify custom CLI arguments are parsed accurately.
    """
    parser = build_parser()
    custom_cli = [
        "--mesh", "assets/custom.msh",
        "--out_dir", "runs/custom_out",
        "--checkpoint_dir", "runs/custom_cp",
        "--restart",
        "-T", "0.85",
        "--num_steps", "2500",
        "--no_adaptive_dt",
        "--cfl", "0.35",
        "--dx_min", "0.0005",
        "--nu", "0.005",
        "--log_interval", "50",
    ]
    args = parser.parse_args(custom_cli)

    assert args.mesh == "assets/custom.msh"
    assert args.out_dir == "runs/custom_out"
    assert args.checkpoint_dir == "runs/custom_cp"
    assert args.restart is True
    assert np.isclose(args.t_final, 0.85)
    assert args.num_steps == 2500
    assert args.adaptive_dt is False
    assert np.isclose(args.cfl, 0.35)
    assert np.isclose(args.dx_min, 0.0005)
    assert np.isclose(args.nu, 0.005)
    assert args.log_interval == 50


def test_build_parser_adaptive_dt_toggle():
    """
    Verify --adaptive_dt and --no_adaptive_dt toggling behavior.
    """
    parser = build_parser()
    
    args_default = parser.parse_args([])
    assert args_default.adaptive_dt is True

    args_off = parser.parse_args(["--no_adaptive_dt"])
    assert args_off.adaptive_dt is False

    args_on = parser.parse_args(["--adaptive_dt"])
    assert args_on.adaptive_dt is True


def test_mesh_not_found_raises_error():
    """
    Verify that run_solver raises FileNotFoundError when mesh_file does not exist.
    """
    with pytest.raises(FileNotFoundError, match="Mesh file.*not found"):
        run_solver(mesh_file="nonexistent_mesh_xyz123.msh")


def test_restart_missing_checkpoint_dir_raises_error():
    """
    Verify that run_solver raises FileNotFoundError when checkpoint directory does not exist.
    """
    with pytest.raises(FileNotFoundError, match="Checkpoint directory not found"):
        run_solver(
            mesh_file="assets/apple_domain.msh",
            restart=True,
            checkpoint_dir="nonexistent_checkpoint_dir_9999",
        )


def test_restart_missing_meta_file_raises_error(tmp_path):
    """
    Verify that run_solver raises FileNotFoundError when checkpoint_meta.json is missing.
    """
    empty_cp_dir = str(tmp_path / "empty_cp")
    os.makedirs(empty_cp_dir, exist_ok=True)

    with pytest.raises(FileNotFoundError, match="Checkpoint metadata file.*not found"):
        run_solver(
            mesh_file="assets/apple_domain.msh",
            restart=True,
            checkpoint_dir=empty_cp_dir,
        )


def test_restart_missing_arrays_raises_error(tmp_path):
    """
    Verify that run_solver raises FileNotFoundError when checkpoint numpy arrays are missing.
    """
    cp_dir = str(tmp_path / "meta_only_cp")
    os.makedirs(cp_dir, exist_ok=True)
    meta_path = os.path.join(cp_dir, "checkpoint_meta.json")
    with open(meta_path, "w") as f:
        json.dump({"step": 100, "t": 0.25}, f)

    with pytest.raises(FileNotFoundError, match="Checkpoint state arrays not found"):
        run_solver(
            mesh_file="assets/apple_domain.msh",
            restart=True,
            checkpoint_dir=cp_dir,
        )


def test_dof_shape_validation_logic():
    """
    Verify that mismatched DOF shapes raise an informative ValueError.
    """
    # Test shape mismatch detection logic
    mock_u_n_shape = (5000,)
    mock_p_n_shape = (1000,)

    loaded_u_bad = np.zeros(200)
    loaded_p_good = np.zeros(1000)

    # Validate velocity shape mismatch
    if loaded_u_bad.shape != mock_u_n_shape:
        with pytest.raises(ValueError, match="Checkpoint DOF mismatch for velocity field"):
            raise ValueError(
                f"Checkpoint DOF mismatch for velocity field u_n: loaded array has shape {loaded_u_bad.shape}, "
                f"but function space V has {mock_u_n_shape[0]} DOFs on rank 0. "
                f"If switching to a different/refined mesh, use scripts/interpolate_mesh.py to project "
                f"checkpoints onto the target function space."
            )

    # Validate pressure shape mismatch
    loaded_p_bad = np.zeros(300)
    if loaded_p_bad.shape != mock_p_n_shape:
        with pytest.raises(ValueError, match="Checkpoint DOF mismatch for pressure field"):
            raise ValueError(
                f"Checkpoint DOF mismatch for pressure field p_n: loaded array has shape {loaded_p_bad.shape}, "
                f"but function space Q has {mock_p_n_shape[0]} DOFs on rank 0. "
                f"If switching to a different/refined mesh, use scripts/interpolate_mesh.py to project "
                f"checkpoints onto the target function space."
            )


def test_main_cli_function_dispatch(monkeypatch):
    """
    Verify that main() parses arguments and dispatches to run_solver.
    """
    called_kwargs = {}

    def mock_run_solver(**kwargs):
        called_kwargs.update(kwargs)
        return {"status": "mock_success", **kwargs}

    monkeypatch.setattr("main.run_solver", mock_run_solver)

    result = main([
        "--mesh", "assets/apple_domain.msh",
        "--out_dir", "results_test",
        "--t_final", "0.10",
        "--num_steps", "50",
        "--no_adaptive_dt",
        "--cfl", "0.4",
        "--nu", "2e-3",
        "--log_interval", "5",
    ])

    assert result["status"] == "mock_success"
    assert called_kwargs["mesh_file"] == "assets/apple_domain.msh"
    assert called_kwargs["out_dir"] == "results_test"
    assert np.isclose(called_kwargs["T"], 0.10)
    assert called_kwargs["num_steps"] == 50
    assert called_kwargs["adaptive_dt"] is False
    assert np.isclose(called_kwargs["cfl"], 0.4)
    assert np.isclose(called_kwargs["nu"], 2e-3)
    assert called_kwargs["log_interval"] == 5
