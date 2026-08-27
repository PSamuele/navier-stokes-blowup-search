"""Shared fixtures for the Run 3 test suite."""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture(scope="session")
def mesh_file(tmp_path_factory):
    """A small but properly graded mesh, generated once for the whole session."""
    from src import mesh

    path = str(tmp_path_factory.mktemp("shared_mesh") / "apple_test.msh")
    mesh.generate_mesh(
        output_file=path, lc_pole=0.02, lc_boundary=0.08, verbosity=0
    )
    return path
