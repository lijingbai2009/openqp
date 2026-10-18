"""conftest.py refuses ``--forked`` in a session that ``import oqp`` puts under MPI.

``oqp.utils.mpi_utils`` calls MPI_Init at import, so with mpi4py installed the
pytest process that pytest-forked forks once per test is already an MPI
process.  On macOS that MPI_Init also starts libdispatch, and Accelerate runs
its threaded BLAS through dispatch_apply, which a forked child cannot use:
75 tests in 28 files died with SIGSEGV inside libBLAS, all of them passing
without --forked.  The refusal turns that into one usage error naming the
cause.  These tests pin both sides of the decision and the import it mirrors.
"""
import builtins
import os
import re

import pytest

import conftest

_MPI_UTILS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "pyoqp", "oqp", "utils", "mpi_utils.py")


class _Config:
    def __init__(self, forked):
        self._forked = forked

    def getoption(self, name, default=None):
        assert name == "forked"
        return self._forked


def test_forked_session_is_refused_when_import_oqp_initialises_mpi(monkeypatch):
    monkeypatch.setattr(conftest, "_import_oqp_initialises_mpi", lambda: True)
    with pytest.raises(pytest.UsageError) as refusal:
        conftest.pytest_configure(_Config(forked=True))
    message = str(refusal.value)
    assert "MPI_Init" in message and "fork()" in message
    assert "dispatch_apply" in message


def test_forked_session_without_mpi_is_allowed(monkeypatch):
    monkeypatch.setattr(conftest, "_import_oqp_initialises_mpi", lambda: False)
    conftest.pytest_configure(_Config(forked=True))


def test_unforked_session_never_probes_mpi(monkeypatch):
    # The probe imports mpi4py.MPI, i.e. calls MPI_Init; an unforked session
    # must not pay for that or have its MPI state touched by the conftest.
    def probe():
        raise AssertionError("MPI probed in an unforked session")
    monkeypatch.setattr(conftest, "_import_oqp_initialises_mpi", probe)
    conftest.pytest_configure(_Config(forked=False))


@pytest.mark.parametrize("error", [ImportError, OSError])
def test_probe_treats_a_missing_mpi_like_mpi_utils_does(monkeypatch, error):
    real_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name == "mpi4py" or name.startswith("mpi4py."):
            raise error("no MPI here")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", failing_import)
    assert conftest._import_oqp_initialises_mpi() is False


def test_probe_mirrors_the_import_in_mpi_utils():
    # If mpi_utils stops initialising MPI at import (or catches other errors),
    # the probe above no longer describes what collection does.
    with open(_MPI_UTILS, encoding="utf-8") as handle:
        source = handle.read()
    assert re.search(r"^try:\n\s+from mpi4py import MPI\n", source, re.M)
    assert re.search(r"^except \(ImportError, OSError\)", source, re.M)
