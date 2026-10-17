"""Tests for oqp.interop.ccpy (external CCpy coupled-cluster library).

The unit tests need neither a compiled OpenQP nor CCpy. The end-to-end cases
run an OpenQP SCF, build the CCpy Hamiltonian from OpenQP's own integrals and
compare with reference energies from the independent PySCF -> CCpy route
(tests/data/ccpy_openqp_reference.json). They run wherever a built OpenQP is
importable; CCpy is then downloaded on demand by ensure_ccpy(), and a failed
download or build fails the test.
"""

import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).parent
DATA = _HERE / "data" / "ccpy_openqp_reference.json"
_MODULE = _HERE.parent / "pyoqp" / "oqp" / "interop" / "ccpy.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("_oqp_interop_ccpy", _MODULE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ccpy_interop = _load_module()


# --------------------------------------------------------------------------
# On-demand installation
# --------------------------------------------------------------------------

def _hide_ccpy(monkeypatch):
    real = ccpy_interop.importlib.import_module
    calls = {"n": 0}

    def fake(name, *args, **kwargs):
        if name == "ccpy":
            calls["n"] += 1
            if calls["n"] == 1:
                raise ImportError("No module named 'ccpy'")
            return types.SimpleNamespace(__name__="ccpy")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(ccpy_interop.importlib, "import_module", fake)
    return calls


def test_ensure_ccpy_disabled_raises_with_pinned_command(monkeypatch):
    _hide_ccpy(monkeypatch)
    monkeypatch.setenv("OPENQP_CCPY_AUTO_INSTALL", "0")
    with pytest.raises(ImportError) as err:
        ccpy_interop.ensure_ccpy()
    assert ccpy_interop.CCPY_COMMIT in str(err.value)


def test_ensure_ccpy_installs_pinned_commit(monkeypatch):
    calls = _hide_ccpy(monkeypatch)
    monkeypatch.delenv("OPENQP_CCPY_AUTO_INSTALL", raising=False)
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(ccpy_interop.subprocess, "run", fake_run)
    mod = ccpy_interop.ensure_ccpy()
    assert mod.__name__ == "ccpy"
    assert calls["n"] == 2
    assert seen["cmd"][:4] == [sys.executable, "-m", "pip", "install"]
    assert seen["cmd"][4] == (f"coupled-cluster-py @ git+{ccpy_interop.CCPY_REPOSITORY}"
                              f"@{ccpy_interop.CCPY_COMMIT}")


def test_ensure_ccpy_pip_failure_raises(monkeypatch):
    _hide_ccpy(monkeypatch)
    monkeypatch.setattr(
        ccpy_interop.subprocess, "run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", "meson: no fortran"))
    with pytest.raises(RuntimeError, match="no fortran"):
        ccpy_interop.ensure_ccpy(install=True)


# --------------------------------------------------------------------------
# UHF reference energy from AO integrals
# --------------------------------------------------------------------------

def _random_eri_chemist(n, rng):
    """(pq|rs) with the 8-fold permutational symmetry of real orbitals."""
    x = rng.standard_normal((n, n, n, n))
    x = x + x.transpose(1, 0, 2, 3)
    x = x + x.transpose(0, 1, 3, 2)
    return x + x.transpose(2, 3, 0, 1)


def test_uhf_energy_matches_mo_basis_formula():
    rng = np.random.default_rng(7)
    n, na, nb = 6, 3, 2
    h = rng.standard_normal((n, n))
    h = h + h.T
    g = _random_eri_chemist(n, rng)
    ca = rng.standard_normal((n, n))          # alpha and beta differ, not orthogonal:
    cb = rng.standard_normal((n, n))          # a swapped index or spin changes E
    v_ao = np.transpose(g, (0, 2, 1, 3))      # physicist, as the adapter passes it

    got = ccpy_interop._uhf_energy(h, v_ao, ca, cb, na, nb)

    def mo(c1, c2):
        return np.einsum("pqrs,pi,qj,rk,sl->ijkl", g, c1, c1, c2, c2)
    haa, hbb = ca.T @ h @ ca, cb.T @ h @ cb
    gaa, gbb, gab = mo(ca, ca), mo(cb, cb), mo(ca, cb)
    oa, ob = range(na), range(nb)
    ref = sum(haa[i, i] for i in oa) + sum(hbb[i, i] for i in ob)
    ref += 0.5 * sum(gaa[i, i, j, j] - gaa[i, j, j, i] for i in oa for j in oa)
    ref += 0.5 * sum(gbb[i, i, j, j] - gbb[i, j, j, i] for i in ob for j in ob)
    ref += sum(gab[i, i, j, j] for i in oa for j in ob)
    assert got == pytest.approx(ref, abs=1e-10, rel=1e-12)


# --------------------------------------------------------------------------
# OpenQP -> CCpy end to end
# --------------------------------------------------------------------------

def _have_native_oqp():
    try:
        import oqp
    except Exception:
        return False
    return hasattr(oqp, "int2e")


native = pytest.mark.skipif(not _have_native_oqp(),
                            reason="native OpenQP library with int2e not importable")

_RUNNER = r"""
import json, sys
case = json.loads(sys.argv[1])
atoms = "\n".join("   %d %.10f %.10f %.10f" % tuple(a) for a in case["atoms"])
open("case.inp", "w").write(
    "[input]\nsystem=\n%s\ncharge=0\nruntype=energy\nbasis=%s\nmethod=hf\n\n"
    "[guess]\ntype=huckel\n\n[scf]\ntype=%s\nmultiplicity=%d\nconv=1.0e-11\nmaxit=200\n"
    % (atoms, case["basis"], case["scf_type"], case["multiplicity"]))
from oqp.pyoqp import Runner
runner = Runner(project="case", input_file="case.inp", log="case.log")
runner.run()
from oqp.interop.ccpy import driver_from_openqp
out = {}
def driver():
    drv = driver_from_openqp(runner.mol, nfrozen=case["nfrozen"])
    drv.options["energy_convergence"] = 1.0e-10
    drv.options["amp_convergence"] = 1.0e-8
    drv.options["maximum_iterations"] = 200
    return drv
drv = driver()
drv.run_cc(method="ccsd")
out["e_ref"] = drv.system.reference_energy
out["e_ccsd"] = drv.system.reference_energy + drv.correlation_energy
if "crcc23" in case["tasks"]:
    drv.run_hbar(method="ccsd")
    drv.run_leftcc(method="left_ccsd")
    drv.run_ccp3(method="crcc23")
    out["e_crcc23_d"] = drv.system.reference_energy + drv.correlation_energy + drv.deltap3[0]["D"]
if "sfeomccsd" in case["tasks"]:
    drv.run_hbar(method="ccsd")
    n = case["nroot"]
    drv.run_guess(method="sfcis", multiplicity=1, roots_per_irrep={"A": n},
                  nact_occupied=0, nact_unoccupied=0, use_symmetry=False)
    drv.run_sfeomcc(method="sfeomccsd", state_index=list(range(n)))
    out["omega_singlets"] = [float(drv.vertical_excitation_energy[i]) for i in range(n)]
print("CCPY_RESULT " + json.dumps(out))
"""


def _cases():
    return json.loads(DATA.read_text())["cases"]


@native
@pytest.mark.parametrize("case", _cases(), ids=lambda c: c["name"])
def test_openqp_orbitals_reproduce_pyscf_ccpy(case, tmp_path):
    env = dict(os.environ, OMP_NUM_THREADS=str(case.get("threads", 2)))
    proc = subprocess.run([sys.executable, "-c", _RUNNER, json.dumps(case)],
                          cwd=tmp_path, capture_output=True, text=True,
                          timeout=3600, env=env)
    lines = [l for l in proc.stdout.splitlines() if l.startswith("CCPY_RESULT ")]
    assert proc.returncode == 0 and lines, proc.stdout[-4000:] + proc.stderr[-4000:]
    got = json.loads(lines[-1].split(" ", 1)[1])
    tol = json.loads(DATA.read_text())["tolerance"]
    ref = case["reference"]
    assert got["e_ref"] == pytest.approx(ref["e_ref"], abs=tol)
    assert got["e_ccsd"] == pytest.approx(ref["e_ccsd"], abs=tol)
    if "e_crcc23_d" in ref:
        assert got["e_crcc23_d"] == pytest.approx(ref["e_crcc23_d"], abs=tol)
    if "omega_singlets" in ref:
        assert got["omega_singlets"] == pytest.approx(ref["omega_singlets"], abs=tol)


def test_dft_orbitals_are_rejected_before_ccpy_is_needed(monkeypatch):
    monkeypatch.setattr(ccpy_interop, "ensure_ccpy",
                        lambda *a, **k: pytest.fail("checked after ensure_ccpy"))
    mol = types.SimpleNamespace(config={"scf": {"type": "rhf"},
                                        "input": {"functional": "bhhlyp"}})
    with pytest.raises(ValueError, match="Hartree-Fock"):
        ccpy_interop.hamiltonian_from_openqp(mol)


def test_unsupported_reference_is_rejected(monkeypatch):
    monkeypatch.setattr(ccpy_interop, "ensure_ccpy",
                        lambda *a, **k: pytest.fail("checked after ensure_ccpy"))
    mol = types.SimpleNamespace(config={"scf": {"type": "rks"}, "input": {}})
    with pytest.raises(ValueError, match="RHF, ROHF or UHF"):
        ccpy_interop.hamiltonian_from_openqp(mol)
