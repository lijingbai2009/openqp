"""Coupled-cluster calculations with the external CCpy library on OpenQP orbitals.

CCpy (https://github.com/piecuch-group/ccpy, GPL-3.0) provides CCSD, CR-CC(2,3),
CC(P;Q), EOM-CC, IP/EA-EOM-CC and spin-flip EOM-CC. It is an external library:
OpenQP does not vendor or redistribute it. :func:`ensure_ccpy` downloads and
builds the pinned commit into the running Python environment the first time it
is needed, unless ``OPENQP_CCPY_AUTO_INSTALL=0``.

The Hamiltonian is built directly from OpenQP's own integrals, so the AO
ordering and normalization are those of the SCF that produced the orbitals:

* ``OQP::Hcore``                    packed lower triangle -> AO core Hamiltonian
* :func:`oqp.library.ints_2e.eri_ao`  AO ERIs ``(mu nu|la si)``, chemist order
* ``OQP::VEC_MO_A`` / ``OQP::VEC_MO_B``  MO coefficients
* ``OQP::E_MO_A`` / ``OQP::E_MO_B``      orbital energies
* ``nelec_A``, ``nelec_B``, ``vnn``       electron counts, nuclear repulsion

CCpy works in physicist order ``<pq|rs> = (pr|qs)``, hence
``transpose(0, 2, 1, 3)``. Point-group symmetry is not passed on (C1). The
two-electron integrals are held in core, as in ``oqp.quantum``: the AO tensor
and the MO tensors each take ``8 * nbf**4`` bytes (UHF keeps three spin
blocks), so the peak is several times that: an estimated 2-3 GB at
``nbf = 100`` (from the formula, not measured).

Usage::

    from oqp.interop.ccpy import driver_from_openqp
    driver = driver_from_openqp(mol, nfrozen=1)    # mol: converged OpenQP SCF
    driver.run_cc(method="ccsd")
    driver.run_hbar(method="ccsd")
    driver.run_leftcc(method="left_ccsd")
    driver.run_ccp3(method="crcc23")
"""
import importlib
import os
import subprocess
import sys

import numpy as np

__all__ = [
    "CCPY_REPOSITORY", "CCPY_COMMIT", "ensure_ccpy",
    "hamiltonian_from_openqp", "driver_from_openqp",
]

CCPY_REPOSITORY = "https://github.com/piecuch-group/ccpy"
#: Commit validated against OpenQP RHF/UHF/ROHF references (2026-10-17).
CCPY_COMMIT = "cf72da4865e054f7a935e2df27e920cf5e335435"

#: The pinned CCpy commit declares requires-python >= 3.10.
CCPY_MIN_PYTHON = (3, 10)

#: Same agreement threshold CCpy's own PySCF interface applies.
REFERENCE_ENERGY_TOLERANCE = 1.0e-6


def _auto_install_enabled():
    value = os.environ.get("OPENQP_CCPY_AUTO_INSTALL", "1").strip().lower()
    return value not in ("0", "false", "no", "off")


def ensure_ccpy(install=None):
    """Import CCpy, downloading and building the pinned commit when it is absent.

    ``install=None`` follows ``OPENQP_CCPY_AUTO_INSTALL`` (default on). The
    build needs git, a Fortran compiler and OpenBLAS (CCpy's meson defaults).
    Returns the imported ``ccpy`` module; raises ``ImportError`` when CCpy is
    absent and installation is disabled, ``RuntimeError`` when pip fails.
    """
    try:
        return importlib.import_module("ccpy")
    except ImportError as exc:
        missing = exc
    if install is None:
        install = _auto_install_enabled()
    spec = f"coupled-cluster-py @ git+{CCPY_REPOSITORY}@{CCPY_COMMIT}"
    if tuple(sys.version_info[:2]) < CCPY_MIN_PYTHON:
        raise ImportError(
            "CCpy needs Python >= %d.%d; this OpenQP runs on Python %d.%d."
            % (CCPY_MIN_PYTHON + tuple(sys.version_info[:2]))) from missing
    if not install:
        raise ImportError(
            "CCpy is not installed and OPENQP_CCPY_AUTO_INSTALL=0. Install it with: "
            f"{sys.executable} -m pip install '{spec}'") from missing

    # Concurrent calculations in one environment must not run two pip builds
    # into the same site-packages; the second waits and then finds CCpy.
    with _install_lock():
        importlib.invalidate_caches()
        try:
            return importlib.import_module("ccpy")
        except ImportError:
            pass
        print(f"   CCpy not found; downloading and building {CCPY_REPOSITORY}"
              f"@{CCPY_COMMIT} (GPL-3.0, external library) into {sys.prefix}. "
              "Set OPENQP_CCPY_AUTO_INSTALL=0 to disable.", flush=True)
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "install", spec],
            capture_output=True, text=True, env=_build_env())
        if proc.returncode != 0:
            tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-40:])
            raise RuntimeError(
                f"pip could not install CCpy (exit {proc.returncode}). The build "
                f"needs git, a Fortran compiler and OpenBLAS.\n{tail}")
        importlib.invalidate_caches()
        return importlib.import_module("ccpy")


#: Homebrew installs OpenBLAS keg-only, so its openblas.pc is not on the
#: default pkg-config path that CCpy's meson build searches.
_HOMEBREW_OPENBLAS = ("/opt/homebrew/opt/openblas", "/usr/local/opt/openblas")


def _build_env():
    """Environment for the CCpy pip build.

    On macOS, append Homebrew's OpenBLAS pkg-config directory (lowest
    priority, so an OpenBLAS the user already exposes still wins).
    """
    env = dict(os.environ)
    if sys.platform != "darwin":
        return env
    for prefix in _HOMEBREW_OPENBLAS:
        pcdir = os.path.join(prefix, "lib", "pkgconfig")
        if os.path.isfile(os.path.join(pcdir, "openblas.pc")):
            paths = [p for p in env.get("PKG_CONFIG_PATH", "").split(os.pathsep) if p]
            if pcdir not in paths:
                env["PKG_CONFIG_PATH"] = os.pathsep.join(paths + [pcdir])
            break
    return env


def _lock_paths():
    """Lock file in the environment, else in the temp directory.

    The temp-directory name is derived from sys.prefix, so processes that
    install into the same environment still share one lock.
    """
    import hashlib
    import tempfile
    tag = hashlib.sha256(os.path.realpath(sys.prefix).encode()).hexdigest()[:16]
    return (os.path.join(sys.prefix, ".openqp-ccpy-install.lock"),
            os.path.join(tempfile.gettempdir(), f"openqp-ccpy-install-{tag}.lock"))


class _install_lock:
    """Exclusive lock serializing the CCpy install for one environment."""

    def __enter__(self):
        errors = []
        for path in _lock_paths():
            try:
                self._fh = open(path, "a+")
                break
            except OSError as exc:
                errors.append(f"{path}: {exc}")
        else:
            raise OSError("cannot create a CCpy install lock: " + "; ".join(errors))
        try:
            if os.name == "nt":
                import msvcrt
                self._fh.seek(0)
                self._unlock = lambda: msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl
                self._unlock = lambda: fcntl.flock(self._fh, fcntl.LOCK_UN)
                fcntl.flock(self._fh, fcntl.LOCK_EX)
        except BaseException:
            self._fh.close()
            raise
        return self

    def __exit__(self, *exc):
        try:
            self._unlock()
        finally:
            self._fh.close()
        return False


def _scalar(mol, key):
    return float(np.asarray(mol.data[key]).ravel()[0])


def _count(mol, key):
    return int(np.asarray(mol.data[key]).ravel()[0])


def _openqp_ao_data(mol):
    from oqp.library.ints_2e import eri_ao
    from oqp.quantum.integrals import unpack_triangular

    basis = mol.data.get_basis()
    if not basis:
        raise RuntimeError("No basis available; run the OpenQP SCF first.")
    nbf = int(basis["nbf"])
    return {
        "nbf": nbf,
        "hcore": unpack_triangular(mol.data["OQP::Hcore"], nbf),
        "v_ao": np.transpose(eri_ao(mol), (0, 2, 1, 3)),     # chemist -> physicist
        "ca": mol._mo_coefficients("OQP::VEC_MO_A", nbf),
        "cb": mol._mo_coefficients("OQP::VEC_MO_B", nbf),
        "ea": np.asarray(mol.data["OQP::E_MO_A"], dtype=float).ravel()[:nbf],
        "eb": np.asarray(mol.data["OQP::E_MO_B"], dtype=float).ravel()[:nbf],
        "na": _count(mol, "nelec_A"),
        "nb": _count(mol, "nelec_B"),
        "vnn": _scalar(mol, "vnn"),
    }


def _uhf_energy(hcore, v_ao, ca, cb, na, nb):
    """Electronic UHF energy of the determinant (ca[:, :na], cb[:, :nb])."""
    da = ca[:, :na] @ ca[:, :na].T
    db = cb[:, :nb] @ cb[:, :nb].T
    dt = da + db
    # v_ao is physicist <mu la|nu si> = (mu nu|la si)
    j = np.einsum("mlns,ls->mn", v_ao, dt, optimize=True)
    ka = np.einsum("mlsn,ls->mn", v_ao, da, optimize=True)
    kb = np.einsum("mlsn,ls->mn", v_ao, db, optimize=True)
    return (np.einsum("mn,mn->", dt, hcore)
            + 0.5 * np.einsum("mn,mn->", dt, j)
            - 0.5 * np.einsum("mn,mn->", da, ka)
            - 0.5 * np.einsum("mn,mn->", db, kb))


def _check_reference(name, e_integrals, e_scf):
    if abs(e_integrals - e_scf) > REFERENCE_ENERGY_TOLERANCE:
        raise RuntimeError(
            f"{name} energy rebuilt from OpenQP integrals ({e_integrals:.10f}) "
            f"differs from the stored OpenQP energy ({e_scf:.10f}). Use the "
            "molecule of a converged method=hf run: a correlated method "
            "overwrites the stored energy, and other orbitals do not match it.")


def _scf_energy(mol):
    if not bool(mol.data._data.mol_energy.SCF_converged):
        raise RuntimeError("The OpenQP SCF is not converged; CCpy needs a "
                           "converged RHF, ROHF or UHF reference.")
    return float(mol.get_scf_energy())


def hamiltonian_from_openqp(mol, nfrozen=0, ndelete=0, reference=None,
                            normal_ordered=True, sorted=True,
                            rohf_canonicalization="Guest-Saunders"):
    """Return ``(system, hamiltonian)`` for ``ccpy.drivers.driver.Driver``.

    ``reference`` defaults to the OpenQP ``[scf] type`` (``rhf``, ``rohf`` or
    ``uhf``). The reference energy is rebuilt from the integrals and must agree
    with the OpenQP SCF energy to ``REFERENCE_ENERGY_TOLERANCE``.
    """
    ref = (reference or mol.config["scf"]["type"]).lower()
    if ref not in ("rhf", "rohf", "uhf"):
        raise ValueError(
            f"CCpy needs an RHF, ROHF or UHF reference; got [scf] type={ref!r}.")
    if mol.config["input"].get("functional"):
        raise ValueError(
            "CCpy needs Hartree-Fock orbitals; the OpenQP SCF used functional="
            f"{mol.config['input']['functional']!r}.")
    if nfrozen < 0 or ndelete < 0:
        raise ValueError(f"nfrozen={nfrozen} and ndelete={ndelete} must be >= 0.")
    ensure_ccpy()
    from ccpy.energy.hf_energy import calc_hf_energy, calc_hf_frozen_core_energy
    from ccpy.interfaces.fcidump_tools import build_rohf_fock
    from ccpy.models.integrals import getHamiltonian, getUHFHamiltonian
    from ccpy.models.system import System, UHFSystem

    d = _openqp_ao_data(mol)
    na, nb, norb = d["na"], d["nb"], d["nbf"]
    if nfrozen > nb:
        raise ValueError(f"nfrozen={nfrozen} exceeds the {nb} doubly occupied "
                         "(beta) orbitals.")
    if ndelete > norb - na:
        raise ValueError(f"ndelete={ndelete} exceeds the {norb - na} virtual "
                         "(alpha) orbitals.")
    nelec, mult = na + nb, na - nb + 1
    e_scf = _scf_energy(mol)

    if ref == "uhf":
        occ = {"a": np.array([1.0] * na + [0.0] * (norb - na)),
               "b": np.array([1.0] * nb + [0.0] * (norb - nb))}
        system = UHFSystem(
            nelec, norb, mult, nfrozen, ndelete=ndelete, point_group="C1",
            orbital_symmetries={"a": ["A"] * norb, "b": ["A"] * norb},
            nuclear_repulsion=d["vnn"],
            mo_energies={"a": d["ea"], "b": d["eb"]}, mo_occupation=occ)
        e_ref = _uhf_energy(d["hcore"], d["v_ao"], d["ca"], d["cb"], na, nb) + d["vnn"]
        _check_reference("UHF", e_ref, e_scf)
        system.reference_energy = e_ref
        hamiltonian = getUHFHamiltonian(d["ca"], d["cb"], d["hcore"], d["v_ao"],
                                        system, normal_ordered, sorted)
        return system, hamiltonian

    c = d["ca"]                       # RHF and ROHF share one set of spatial MOs
    e1int = np.asfortranarray(c.T @ d["hcore"] @ c)
    e2int = np.asfortranarray(np.einsum("ijkl,ip,jq,kr,ls->pqrs", d["v_ao"], c, c, c, c,
                                        optimize=True))
    system = System(nelec, norb, mult, nfrozen, ndelete=ndelete, point_group="C1",
                    orbital_symmetries=["A"] * norb, nuclear_repulsion=d["vnn"],
                    mo_energies=d["ea"])
    if ref == "rohf":
        system.mo_energies, _ = build_rohf_fock(e1int, e2int, system,
                                                rohf_canonicalization)
    e_ref = calc_hf_energy(e1int, e2int, system) + d["vnn"]
    _check_reference(ref.upper(), e_ref, e_scf)
    system.reference_energy = e_ref
    system.frozen_energy = calc_hf_frozen_core_energy(e1int, e2int, system)
    return system, getHamiltonian(e1int, e2int, system, normal_ordered, sorted)


def driver_from_openqp(mol, nfrozen=0, ndelete=0, reference=None, **kwargs):
    """Return a ``ccpy.drivers.driver.Driver`` for a converged OpenQP SCF.

    For a UHF reference the driver runs with ``RHF_symmetry=False``. Build the
    ``Driver`` through this function rather than from
    :func:`hamiltonian_from_openqp` directly, or set that option yourself.
    """
    ref = (reference or mol.config["scf"]["type"]).lower()
    # hamiltonian_from_openqp validates the request before it installs CCpy,
    # so an invalid request never downloads anything.
    system, hamiltonian = hamiltonian_from_openqp(
        mol, nfrozen=nfrozen, ndelete=ndelete, reference=reference, **kwargs)
    from ccpy.drivers.driver import Driver
    driver = Driver(system, hamiltonian)
    if ref == "uhf":
        # CCpy turns RHF_symmetry on whenever the alpha and beta occupied
        # counts are equal and then copies alpha amplitudes into beta ones.
        # A UHF singlet (broken symmetry) has different alpha and beta
        # orbitals, so that shortcut would be wrong.
        driver.options["RHF_symmetry"] = False
    return driver
