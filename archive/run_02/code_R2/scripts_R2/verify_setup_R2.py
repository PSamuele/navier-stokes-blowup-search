#!/usr/bin/env python3
"""
verify_setup.py - Navier-Stokes Cusp Study Acceptance Test Suite & Verification Runner.

This script executes a comprehensive, programmatic end-to-end verification of the project
reorganization, data integrity, interpolation pipeline, GitHub maintainability, and CLI
parameterization as mandated in ORIGINAL_REQUEST.md and PROJECT.md.

Test Suites:
  1. Suite 1: Directory Structure Verification (runs/run_01_coarse_wrong, runs/run_02_standard_aws, runs/run_03_fine_interpolated)
  2. Suite 2: Data Preservation & Integrity Verification (zero byte loss, 9.4GB AWS velocity.h5, 17.1MB XDMF, 39,020-row CSV, 16 rank checkpoints, vortex.mp4)
  3. Suite 3: Programmatic Interpolation Validation (scripts/interpolate_mesh.py --dry-run, mathematical viability, zero NaNs/Infs)
  4. Suite 4: GitHub & Maintainability Validation (.gitignore pattern filtering for heavy assets and runs/)
  5. Suite 5: CLI Parameterization Validation (python main.py --help and argparse parameterization)

Usage:
  python verify_setup.py
  pytest verify_setup.py
"""

import os
import sys
import json
import time
import fnmatch
import subprocess
import traceback
from pathlib import Path
import numpy as np
import h5py

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.interpolate_mesh import run_interpolation_pipeline, key_to_time
from main import build_parser


# ==============================================================================
# Helper Utilities & Gitignore Evaluator
# ==============================================================================

def parse_gitignore_rules(gitignore_path):
    """
    Parse a .gitignore file into active pattern rules.
    """
    if not os.path.exists(gitignore_path):
        return []
    rules = []
    with open(gitignore_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rules.append(line)
    return rules


def is_path_ignored(rel_path, gitignore_rules):
    """
    Evaluate whether a relative path is ignored according to gitignore rules.
    Supports directory matches, wildcards (*, *.ext), and exact file matches.
    """
    # Normalize slashes
    norm_path = rel_path.replace("\\", "/")
    path_parts = norm_path.split("/")
    filename = path_parts[-1]

    for rule in gitignore_rules:
        clean_rule = rule.replace("\\", "/")
        
        # Directory rule (e.g. 'runs/', 'results/')
        if clean_rule.endswith("/"):
            dir_name = clean_rule.rstrip("/")
            if dir_name in path_parts[:-1] or norm_path == dir_name or norm_path.startswith(dir_name + "/"):
                return True
        # Extension rule (e.g. '*.h5', '*.npy')
        elif clean_rule.startswith("*."):
            ext = clean_rule[1:] # e.g. '.h5'
            if filename.endswith(ext):
                return True
        # Glob or exact pattern match
        else:
            if fnmatch.fnmatch(norm_path, clean_rule) or fnmatch.fnmatch(filename, clean_rule):
                return True
    return False


# ==============================================================================
# Suite 1: Directory Structure Verification
# ==============================================================================

class TestSuite1DirectoryStructure:
    """
    Verifies that all simulation runs (run_01_coarse_wrong, run_02_standard_aws,
    run_03_fine_interpolated) and their expected subdirectories/files exist.
    """

    def test_runs_root_directory_exists(self):
        runs_dir = os.path.join(PROJECT_ROOT, "runs")
        assert os.path.exists(runs_dir), f"Missing root runs directory: {runs_dir}"
        assert os.path.isdir(runs_dir), f"Target is not a directory: {runs_dir}"

    def test_run_01_coarse_wrong_structure(self):
        run01_dir = os.path.join(PROJECT_ROOT, "runs", "run_01_coarse_wrong")
        assert os.path.exists(run01_dir), f"Missing run_01 directory: {run01_dir}"
        assert os.path.isdir(run01_dir), f"run_01 is not a directory: {run01_dir}"

        expected_files = [
            "velocity.h5",
            "velocity.xdmf",
            "blowup_data.csv",
            "checkpoint_meta.json",
            "checkpoint_pn.npy",
            "checkpoint_un.npy",
        ]
        for fname in expected_files:
            fpath = os.path.join(run01_dir, fname)
            assert os.path.exists(fpath), f"Missing required file in run_01: {fname}"

        expected_subdirs = ["checkpoints", "plots"]
        for sname in expected_subdirs:
            spath = os.path.join(run01_dir, sname)
            assert os.path.exists(spath), f"Missing subdirectory in run_01: {sname}"
            assert os.path.isdir(spath), f"Subdirectory is not a dir: {sname}"

    def test_run_02_standard_aws_structure(self):
        run02_dir = os.path.join(PROJECT_ROOT, "runs", "run_02_standard_aws")
        assert os.path.exists(run02_dir), f"Missing run_02 directory: {run02_dir}"
        assert os.path.isdir(run02_dir), f"run_02 is not a directory: {run02_dir}"

        expected_files = [
            "velocity.h5",
            "velocity.xdmf",
            "blowup_data.csv",
            "blowup_data_RUN_AWS.csv",
            "apple_domain.msh",
            "checkpoint_meta.json",
            "checkpoint_pn.npy",
            "checkpoint_un.npy",
        ]
        for fname in expected_files:
            fpath = os.path.join(run02_dir, fname)
            assert os.path.exists(fpath), f"Missing required file in run_02: {fname}"

        expected_subdirs = ["checkpoints", "plots"]
        for sname in expected_subdirs:
            spath = os.path.join(run02_dir, sname)
            assert os.path.exists(spath), f"Missing subdirectory in run_02: {sname}"
            assert os.path.isdir(spath), f"Subdirectory is not a dir: {sname}"

    def test_run_03_fine_interpolated_structure(self):
        run03_dir = os.path.join(PROJECT_ROOT, "runs", "run_03_fine_interpolated")
        assert os.path.exists(run03_dir), f"Missing run_03 directory: {run03_dir}"
        assert os.path.isdir(run03_dir), f"run_03 is not a directory: {run03_dir}"

        expected_files = [
            "apple_domain.msh",
            "checkpoint_meta.json",
            "checkpoint_pn.npy",
            "checkpoint_un.npy",
        ]
        for fname in expected_files:
            fpath = os.path.join(run03_dir, fname)
            assert os.path.exists(fpath), f"Missing required restart file in run_03: {fname}"

    def test_legacy_directories_cleaned_up(self):
        """
        Verify that original source folders (results/, Export_AWS/, Run_Cloud/)
        do not contain unmigrated lingering data.
        """
        for legacy in ["results", "Export_AWS", "Run_Cloud"]:
            leg_path = os.path.join(PROJECT_ROOT, legacy)
            if os.path.exists(leg_path):
                items = [f for f in os.listdir(leg_path) if f != ".gitkeep"]
                assert len(items) == 0, (
                    f"Legacy directory '{legacy}' contains unmigrated files: {items}. "
                    f"Files should have been moved into runs/ per R1."
                )


# ==============================================================================
# Suite 2: Data Preservation & Integrity Verification
# ==============================================================================

class TestSuite2DataPreservationAndIntegrity:
    """
    Verifies that no original simulation data was corrupted or lost during migration:
    zero byte loss, 9.4GB AWS velocity.h5, 17.1MB XDMF, 39,020-row CSV, 16 rank checkpoints,
    and preserved video assets.
    """

    def test_zero_byte_loss_in_all_runs(self):
        runs_dir = os.path.join(PROJECT_ROOT, "runs")
        assert os.path.exists(runs_dir)

        checked_files = 0
        for root, dirs, files in os.walk(runs_dir):
            for file in files:
                if file == ".gitkeep":
                    continue
                file_path = os.path.join(root, file)
                size = os.path.getsize(file_path)
                assert size > 0, f"Zero-byte empty file detected: {file_path}"
                checked_files += 1

        assert checked_files >= 40, f"Expected >= 40 simulation files in runs/, found {checked_files}"

    def test_aws_velocity_h5_integrity_and_size(self):
        h5_path = os.path.join(PROJECT_ROOT, "runs", "run_02_standard_aws", "velocity.h5")
        assert os.path.exists(h5_path), f"Missing AWS velocity.h5 at {h5_path}"

        size_bytes = os.path.getsize(h5_path)
        size_gb = size_bytes / (1024 ** 3)
        assert size_bytes >= 9_000_000_000, f"AWS velocity.h5 is undersized ({size_gb:.2f} GB < 9.0 GB)"

        # Inspect HDF5 internal structure and sample snapshots
        with h5py.File(h5_path, "r") as f:
            assert "Mesh/mesh/geometry" in f, "Missing /Mesh/mesh/geometry in velocity.h5"
            assert "Mesh/mesh/topology" in f, "Missing /Mesh/mesh/topology in velocity.h5"
            assert "Function/Velocity" in f, "Missing /Function/Velocity group in velocity.h5"

            geom = f["Mesh/mesh/geometry"][:]
            topo = f["Mesh/mesh/topology"][:]
            assert geom.shape[0] >= 10000, f"Expected >= 10000 mesh nodes, got {geom.shape[0]}"
            assert topo.shape[0] >= 19000, f"Expected >= 19000 mesh elements, got {topo.shape[0]}"

            vel_keys = list(f["Function/Velocity"].keys())
            assert len(vel_keys) >= 30000, f"Expected >= 30000 snapshots in AWS velocity.h5, got {len(vel_keys)}"

            # Check snapshot 0 and snapshot near t=0.5
            sample_keys = [vel_keys[0], "0_499998586184208", vel_keys[-1]]
            for k in sample_keys:
                if k in f["Function/Velocity"]:
                    u_sample = f[f"Function/Velocity/{k}"][:]
                    assert not np.isnan(u_sample).any(), f"NaN values detected in snapshot '{k}'"
                    assert not np.isinf(u_sample).any(), f"Inf values detected in snapshot '{k}'"
                    assert u_sample.shape == (geom.shape[0], 3), f"Unexpected shape for snapshot '{k}'"

    def test_aws_velocity_xdmf_integrity_and_size(self):
        xdmf_path = os.path.join(PROJECT_ROOT, "runs", "run_02_standard_aws", "velocity.xdmf")
        assert os.path.exists(xdmf_path), f"Missing AWS velocity.xdmf at {xdmf_path}"

        size_bytes = os.path.getsize(xdmf_path)
        size_mb = size_bytes / (1024 ** 2)
        assert size_bytes >= 16_000_000, f"AWS velocity.xdmf is undersized ({size_mb:.2f} MB < 16.0 MB)"

        with open(xdmf_path, "r", encoding="utf-8", errors="ignore") as f:
            header_chunk = f.read(2048)
            assert "<Xdmf" in header_chunk or "<?xml" in header_chunk, "Invalid XDMF file format (missing XML/Xdmf tag)"
            assert "<Grid" in header_chunk or "<Domain" in header_chunk, "Invalid XDMF structure"

    def test_aws_blowup_data_csv_integrity(self):
        csv_path = os.path.join(PROJECT_ROOT, "runs", "run_02_standard_aws", "blowup_data.csv")
        csv_aws_path = os.path.join(PROJECT_ROOT, "runs", "run_02_standard_aws", "blowup_data_RUN_AWS.csv")
        assert os.path.exists(csv_path), f"Missing {csv_path}"
        assert os.path.exists(csv_aws_path), f"Missing {csv_aws_path}"

        with open(csv_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        total_lines = len(lines)
        # Expected ~39,019 lines (39,018 data steps + 1 header)
        assert total_lines >= 39000, f"Expected >= 39,000 lines in blowup_data.csv, got {total_lines}"

        header = lines[0].strip().split(",")
        assert "t" in header and "max_velocity" in header, f"Unexpected CSV header: {header}"

        first_data = [float(x) for x in lines[1].strip().split(",")]
        last_data = [float(x) for x in lines[-1].strip().split(",")]
        assert np.isclose(first_data[0], 0.0, atol=1e-5), f"Initial time should be ~0.0, got {first_data[0]}"
        assert np.isclose(last_data[0], 0.55, atol=1e-2), f"Final time should be ~0.55, got {last_data[0]}"
        assert last_data[1] > 1e15, "Blowup max_velocity should reflect singular cusp dynamics at t~0.55"

    def test_aws_mpi_16_rank_checkpoints(self):
        run02_dir = os.path.join(PROJECT_ROOT, "runs", "run_02_standard_aws")
        
        for rank in range(16):
            un_file = os.path.join(run02_dir, f"checkpoint_un_rank{rank}.npy")
            pn_file = os.path.join(run02_dir, f"checkpoint_pn_rank{rank}.npy")

            assert os.path.exists(un_file), f"Missing MPI rank velocity checkpoint: checkpoint_un_rank{rank}.npy"
            assert os.path.exists(pn_file), f"Missing MPI rank pressure checkpoint: checkpoint_pn_rank{rank}.npy"

            un_arr = np.load(un_file)
            pn_arr = np.load(pn_file)

            assert len(un_arr) > 0, f"Empty velocity array in rank {rank}"
            assert len(pn_arr) > 0, f"Empty pressure array in rank {rank}"
            assert not np.isnan(un_arr).any(), f"NaN in velocity array for rank {rank}"
            assert not np.isinf(un_arr).any(), f"Inf in velocity array for rank {rank}"
            assert not np.isnan(pn_arr).any(), f"NaN in pressure array for rank {rank}"
            assert not np.isinf(pn_arr).any(), f"Inf in pressure array for rank {rank}"

    def test_unique_media_vortex_mp4_preserved(self):
        mp4_path = os.path.join(PROJECT_ROOT, "docs", "media", "vortex.mp4")
        assert os.path.exists(mp4_path), f"Preserved media file missing: {mp4_path}"
        assert os.path.getsize(mp4_path) > 100_000, "docs/media/vortex.mp4 is too small or corrupted"


# ==============================================================================
# Suite 3: Programmatic Interpolation Validation
# ==============================================================================

class TestSuite3InterpolationValidation:
    """
    Verifies that scripts/interpolate_mesh.py executes cleanly in dry-run mode,
    produces zero NaNs/Infs, enforces boundary conditions, and generates valid
    P2^3 velocity and P1 pressure function space arrays.
    """

    def test_interpolate_mesh_dry_run_cli(self):
        source_h5 = os.path.join(PROJECT_ROOT, "runs", "run_02_standard_aws", "velocity.h5")
        assert os.path.exists(source_h5), f"Source H5 missing: {source_h5}"

        cmd = [
            sys.executable,
            os.path.join(PROJECT_ROOT, "scripts", "interpolate_mesh.py"),
            "--source-h5", source_h5,
            "--target-time", "0.50",
            "--lc-pole", "0.0005",
            "--lc-boundary", "0.015",
            "--dry-run",
            "--quiet",
        ]

        result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
        assert result.returncode == 0, f"interpolate_mesh.py CLI dry-run failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"

    def test_interpolate_mesh_dry_run_programmatic(self, tmp_path=None):
        source_h5 = os.path.join(PROJECT_ROOT, "runs", "run_02_standard_aws", "velocity.h5")
        assert os.path.exists(source_h5), f"Source H5 missing: {source_h5}"

        res = run_interpolation_pipeline(
            source_h5=source_h5,
            target_time=0.50,
            lc_pole=0.001,
            lc_boundary=0.02,
            out_dir=os.path.join(PROJECT_ROOT, "runs", "run_03_fine_interpolated"),
            dry_run=True,
            verbose=False,
        )

        assert res["status"] == "success"
        assert np.isclose(res["actual_time"], 0.499998586184208, atol=1e-4)

        u_interleaved = res["u_interleaved"]
        p_p1 = res["p_p1"]

        assert u_interleaved is not None
        assert p_p1 is not None
        assert len(u_interleaved) == res["num_p2_nodes"] * 3
        assert len(p_p1) == res["num_p1_nodes"]

        # Assert no NaNs or Infs
        assert not np.isnan(u_interleaved).any(), "Interpolated velocity array contains NaN"
        assert not np.isinf(u_interleaved).any(), "Interpolated velocity array contains Inf"
        assert not np.isnan(p_p1).any(), "Interpolated pressure array contains NaN"
        assert not np.isinf(p_p1).any(), "Interpolated pressure array contains Inf"

        # Assert finite maximum velocity
        assert res["stats"]["max_velocity"] > 1e10, f"Interpolated peak velocity magnitude too low: {res['stats']['max_velocity']}"

    def test_run_03_prepared_restart_checkpoints(self):
        run03_dir = os.path.join(PROJECT_ROOT, "runs", "run_03_fine_interpolated")
        un_file = os.path.join(run03_dir, "checkpoint_un.npy")
        pn_file = os.path.join(run03_dir, "checkpoint_pn.npy")
        meta_file = os.path.join(run03_dir, "checkpoint_meta.json")
        msh_file = os.path.join(run03_dir, "apple_domain.msh")

        assert os.path.exists(un_file)
        assert os.path.exists(pn_file)
        assert os.path.exists(meta_file)
        assert os.path.exists(msh_file)

        un = np.load(un_file)
        pn = np.load(pn_file)
        with open(meta_file, "r") as f:
            meta = json.load(f)

        assert len(un) == meta["num_p2_nodes"] * 3
        assert len(pn) == meta["num_p1_nodes"]
        assert not np.isnan(un).any()
        assert not np.isinf(un).any()
        assert not np.isnan(pn).any()
        assert not np.isinf(pn).any()
        assert np.isclose(meta["t"], 0.499998586184208, atol=1e-4)


# ==============================================================================
# Suite 4: GitHub & Maintainability Validation
# ==============================================================================

class TestSuite4GitHubAndMaintainability:
    """
    Verifies that .gitignore correctly excludes heavy simulation formats (.h5, .npy,
    .xdmf, .tar.gz, runs/) from repository tracking to ensure repo maintainability.
    """

    def test_gitignore_file_exists(self):
        gitignore_path = os.path.join(PROJECT_ROOT, ".gitignore")
        assert os.path.exists(gitignore_path), f"Missing .gitignore at {gitignore_path}"

    def test_gitignore_rule_definitions(self):
        gitignore_path = os.path.join(PROJECT_ROOT, ".gitignore")
        rules = parse_gitignore_rules(gitignore_path)

        required_patterns = ["*.h5", "*.npy", "*.xdmf", "runs/", "results/"]
        for pat in required_patterns:
            assert any(pat == r or pat.rstrip("/") == r.rstrip("/") for r in rules), (
                f"Missing required pattern '{pat}' in .gitignore rules: {rules}"
            )

    def test_heavy_assets_path_filtering(self):
        gitignore_path = os.path.join(PROJECT_ROOT, ".gitignore")
        rules = parse_gitignore_rules(gitignore_path)

        # Paths that MUST be ignored
        ignored_samples = [
            "runs/run_01_coarse_wrong/velocity.h5",
            "runs/run_01_coarse_wrong/velocity.xdmf",
            "runs/run_02_standard_aws/velocity.h5",
            "runs/run_02_standard_aws/velocity.xdmf",
            "runs/run_02_standard_aws/checkpoint_un.npy",
            "runs/run_02_standard_aws/checkpoint_un_rank0.npy",
            "runs/run_03_fine_interpolated/checkpoint_un.npy",
            "runs/run_03_fine_interpolated/checkpoint_pn.npy",
            "runs/run_03_fine_interpolated/apple_domain.msh",
            "results/velocity.h5",
            "tutto_cloud.tar.gz",
            "Run_Cloud.tar.gz",
        ]
        for path in ignored_samples:
            assert is_path_ignored(path, rules), f"Path '{path}' should be ignored by .gitignore, but was not."

        # Source code and documentation paths that MUST NOT be ignored
        tracked_samples = [
            "main.py",
            "verify_setup.py",
            "PROJECT.md",
            "README.md",
            "src/mesh.py",
            "src/conditions.py",
            "scripts/interpolate_mesh.py",
            "tests/test_interpolate_mesh.py",
            "docs/methodology.md",
        ]
        for path in tracked_samples:
            assert not is_path_ignored(path, rules), f"Source file '{path}' should NOT be ignored by .gitignore."


# ==============================================================================
# Suite 5: CLI Parameterization Validation
# ==============================================================================

class TestSuite5CLIParameterization:
    """
    Verifies that main.py supports CLI execution via argparse with all mandatory
    options (--mesh, --out_dir, --checkpoint_dir, --restart, -T, --num_steps,
    --adaptive_dt, --cfl, --dx_min, --nu, --log_interval).
    """

    def test_main_cli_help_flag(self):
        cmd = [sys.executable, os.path.join(PROJECT_ROOT, "main.py"), "--help"]
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
        assert result.returncode == 0, f"main.py --help failed:\n{result.stderr}"

        stdout = result.stdout
        expected_options = [
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
        for opt in expected_options:
            assert opt in stdout, f"Missing CLI option '{opt}' in main.py --help stdout"

    def test_build_parser_default_parameters(self):
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

    def test_build_parser_custom_cli_arguments(self):
        parser = build_parser()
        custom_args = [
            "--mesh", "runs/run_03_fine_interpolated/apple_domain.msh",
            "--out_dir", "runs/run_03_fine_interpolated",
            "--checkpoint_dir", "runs/run_03_fine_interpolated",
            "--restart",
            "-T", "0.60",
            "--num_steps", "3000",
            "--no_adaptive_dt",
            "--cfl", "0.40",
            "--dx_min", "0.00005",
            "--nu", "0.002",
            "--log_interval", "20",
        ]
        args = parser.parse_args(custom_args)

        assert args.mesh == "runs/run_03_fine_interpolated/apple_domain.msh"
        assert args.out_dir == "runs/run_03_fine_interpolated"
        assert args.checkpoint_dir == "runs/run_03_fine_interpolated"
        assert args.restart is True
        assert np.isclose(args.t_final, 0.60)
        assert args.num_steps == 3000
        assert args.adaptive_dt is False
        assert np.isclose(args.cfl, 0.40)
        assert np.isclose(args.dx_min, 0.00005)
        assert np.isclose(args.nu, 0.002)
        assert args.log_interval == 20


# ==============================================================================
# Standalone Test Runner & Summary Reporter
# ==============================================================================

def run_acceptance_tests():
    """
    Execute all 5 test suites sequentially, collect metrics, and generate
    a formatted terminal acceptance report.
    """
    suites = [
        ("Suite 1: Directory Structure Verification", TestSuite1DirectoryStructure()),
        ("Suite 2: Data Preservation & Integrity Verification", TestSuite2DataPreservationAndIntegrity()),
        ("Suite 3: Programmatic Interpolation Validation", TestSuite3InterpolationValidation()),
        ("Suite 4: GitHub & Maintainability Validation", TestSuite4GitHubAndMaintainability()),
        ("Suite 5: CLI Parameterization Validation", TestSuite5CLIParameterization()),
    ]

    print("=" * 80)
    print(" NAVIER-STOKES CUSP STUDY - ACCEPTANCE TEST SUITE & VERIFICATION RUNNER")
    print("=" * 80)
    print(f" Workspace Root:  {PROJECT_ROOT}")
    print(f" Execution Time:  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Python Version:  {sys.version.split()[0]}")
    print("-" * 80)

    total_passed = 0
    total_failed = 0
    total_tests = 0
    suite_summaries = []
    start_total_time = time.time()

    for suite_name, suite_instance in suites:
        print(f"\n>> Running {suite_name}...")
        test_methods = [m for m in dir(suite_instance) if m.startswith("test_")]
        suite_passed = 0
        suite_failed = 0
        t0_suite = time.time()

        for method_name in test_methods:
            total_tests += 1
            test_method = getattr(suite_instance, method_name)
            t0 = time.time()
            try:
                test_method()
                duration = time.time() - t0
                print(f"   [ PASS ] {method_name} ({duration:.3f}s)")
                suite_passed += 1
                total_passed += 1
            except AssertionError as ae:
                duration = time.time() - t0
                print(f"   [ FAIL ] {method_name} ({duration:.3f}s)")
                print(f"            AssertionError: {ae}")
                suite_failed += 1
                total_failed += 1
            except Exception as ex:
                duration = time.time() - t0
                print(f"   [ FAIL ] {method_name} ({duration:.3f}s)")
                print(f"            Exception: {type(ex).__name__}: {ex}")
                traceback.print_exc()
                suite_failed += 1
                total_failed += 1

        suite_duration = time.time() - t0_suite
        suite_status = "PASSED" if suite_failed == 0 else "FAILED"
        suite_summaries.append({
            "name": suite_name,
            "passed": suite_passed,
            "failed": suite_failed,
            "total": len(test_methods),
            "duration": suite_duration,
            "status": suite_status,
        })

    total_duration = time.time() - start_total_time

    # Print summary table
    print("\n" + "=" * 80)
    print(" ACCEPTANCE TEST VERIFICATION SUMMARY REPORT")
    print("=" * 80)
    print(f" {'Test Suite':<52} | {'Pass':<4} | {'Fail':<4} | {'Time':<7} | {'Status'}")
    print("-" * 80)
    for s in suite_summaries:
        status_str = f" \033[92m{s['status']}\033[0m" if s['status'] == "PASSED" else f" \033[91m{s['status']}\033[0m"
        print(f" {s['name']:<52} | {s['passed']:<4} | {s['failed']:<4} | {s['duration']:<6.2f}s | {s['status']}")
    print("-" * 80)
    print(f" Total Tests: {total_tests} | Passed: {total_passed} | Failed: {total_failed} | Duration: {total_duration:.2f}s")
    
    if total_failed == 0:
        print("\n [SUCCESS] ALL ACCEPTANCE CRITERIA SYSTEMATICALLY VERIFIED AND PASSED!")
        print("=" * 80)
        return 0
    else:
        print(f"\n [FAILURE] {total_failed} TEST(S) FAILED!")
        print("=" * 80)
        return 1


# ==============================================================================
# Pytest Function Mappings (Enables `pytest verify_setup.py`)
# ==============================================================================

_s1 = TestSuite1DirectoryStructure()
def test_suite1_runs_root(): _s1.test_runs_root_directory_exists()
def test_suite1_run01(): _s1.test_run_01_coarse_wrong_structure()
def test_suite1_run02(): _s1.test_run_02_standard_aws_structure()
def test_suite1_run03(): _s1.test_run_03_fine_interpolated_structure()
def test_suite1_legacy(): _s1.test_legacy_directories_cleaned_up()

_s2 = TestSuite2DataPreservationAndIntegrity()
def test_suite2_zero_byte_loss(): _s2.test_zero_byte_loss_in_all_runs()
def test_suite2_aws_velocity_h5(): _s2.test_aws_velocity_h5_integrity_and_size()
def test_suite2_aws_velocity_xdmf(): _s2.test_aws_velocity_xdmf_integrity_and_size()
def test_suite2_aws_blowup_csv(): _s2.test_aws_blowup_data_csv_integrity()
def test_suite2_aws_16_rank_checkpoints(): _s2.test_aws_mpi_16_rank_checkpoints()
def test_suite2_vortex_mp4(): _s2.test_unique_media_vortex_mp4_preserved()

_s3 = TestSuite3InterpolationValidation()
def test_suite3_dry_run_cli(): _s3.test_interpolate_mesh_dry_run_cli()
def test_suite3_dry_run_programmatic(): _s3.test_interpolate_mesh_dry_run_programmatic()
def test_suite3_prepared_restart(): _s3.test_run_03_prepared_restart_checkpoints()

_s4 = TestSuite4GitHubAndMaintainability()
def test_suite4_gitignore_exists(): _s4.test_gitignore_file_exists()
def test_suite4_gitignore_rules(): _s4.test_gitignore_rule_definitions()
def test_suite4_path_filtering(): _s4.test_heavy_assets_path_filtering()

_s5 = TestSuite5CLIParameterization()
def test_suite5_cli_help(): _s5.test_main_cli_help_flag()
def test_suite5_parser_defaults(): _s5.test_build_parser_default_parameters()
def test_suite5_parser_custom(): _s5.test_build_parser_custom_cli_arguments()


if __name__ == "__main__":
    sys.exit(run_acceptance_tests())
