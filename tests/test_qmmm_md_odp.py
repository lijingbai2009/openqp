"""ODP umbrella, droplet and solute-COM restraints in ground-state QM/MM MD."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from oqp_module_isolation import child_env

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples" / "QMMM"


def test_parser_and_checker_accept_bias_only_with_qmmm_md():
    from oqp.utils.input_checker import CheckReport, _check_runtype
    from oqp.utils.oqp_input import OQPInputError, parse_canonical_oqp

    deck = ('hf/sto-3g md(nstep=2,ensemble=nvt) qmmm(qm_atoms="0-3") '
            'droplet(enabled=true,radius=6.0) solute_com(enabled=true) '
            'odp(enabled=true,cv="distance(1,5)",scale="1",reference_r="5",'
            'reference_p="6",k_parallel=0.1) geom="w.pdb"')
    spec = parse_canonical_oqp(deck)
    assert {call.name for call in spec.modifiers} >= {"odp", "droplet", "solute_com"}
    with pytest.raises(OQPInputError, match="QM/MM md"):
        parse_canonical_oqp('hf/sto-3g md(nstep=2) odp(enabled=true,'
                            'k_parallel=0.1) geom="w.xyz"')
    with pytest.raises(OQPInputError, match="only to namd"):
        parse_canonical_oqp('hf/sto-3g md(nstep=2) qmmm(qm_atoms="0-3") '
                            'flexible(enabled=true,alpha=5,temperature=300) '
                            'geom="w.pdb"')
    for qmmm_flag, expected in ((True, False), (False, True)):
        report = CheckReport()
        _check_runtype({"input": {"runtype": "md", "qmmm_flag": qmmm_flag},
                        "odp": {"enabled": True},
                        "droplet": {"enabled": True}}, report)
        paths = {item.path for item in report.errors}
        assert ("odp.enabled" in paths) is expected
        assert ("droplet.enabled" in paths) is expected


BIAS_SCRIPT = r"""
import json
import numpy as np
from openmm import app
from oqp.library.qmmm_md_bias import QMMMMDBias

pdb = app.PDBFile(EXAMPLES + "/formaldehyde_water.pdb")
natom = pdb.topology.getNumAtoms()
masses = np.array([12.011, 15.999, 1.008, 1.008] + [15.999, 1.008, 1.008]*((natom - 4)//3))
masses[6] = 0.0          # a massless site (as in TIP4P/TIP5P) inside a water
sections = {
    "odp": {"enabled": True, "cv": "distance(1,8); asymmetric_distance(2,9,1,12)",
            "scale": [0.8, 1.1], "reference_r": [6.5, 1.0],
            "reference_p": [5.0, -1.5], "center": 0.35, "k_parallel": 0.07,
            "k_perpendicular": 0.02, "window": 3},
    "droplet": {"enabled": True, "radius": 3.0, "buffer": 0.5,
                "force_constant": 4.0, "max_penetration": 5.0},
    "solute_com": {"enabled": True, "center": "0.3,-0.2,0.4",
                   "force_constant": 2.0, "atoms": "0-3"},
}
bias = QMMMMDBias(sections, pdb.topology, list(range(4)), masses, False)
rng = np.random.default_rng(5)
xyz = np.array(pdb.positions.value_in_unit(pdb.positions.unit))*10/0.52917721090
xyz = xyz + rng.normal(scale=0.15, size=xyz.shape)
energy, force, record = bias.evaluate(xyz)
h = 1.0e-5
worst = 0.0
for atom in range(natom):
    for axis in range(3):
        plus = xyz.copy(); plus[atom, axis] += h
        minus = xyz.copy(); minus[atom, axis] -= h
        fd = -(bias.evaluate(plus)[0] - bias.evaluate(minus)[0])/(2*h)
        worst = max(worst, abs(fd - force[atom, axis]))
print("BIAS=" + json.dumps({
    "energy": energy,
    "parts": [record["odp_energy_hartree"], record["droplet_energy_hartree"],
              record["solute_com_energy_hartree"]],
    "fd": worst,
    "massless_force": float(np.abs(force[6]).max()),
    "settings": bias.settings(),
}))
"""


def test_bias_force_is_the_gradient_of_its_energy():
    pytest.importorskip("openmm")
    script = BIAS_SCRIPT.replace("EXAMPLES", repr(str(EXAMPLES)))
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT,
                            env=child_env(), capture_output=True, text=True,
                            check=False)
    if result.returncode != 0:
        if "cannot load" in result.stderr or "No module named" in result.stderr:
            pytest.skip("compiled OpenQP runtime is not available")
        pytest.fail(result.stdout + result.stderr)
    marker = next(l for l in result.stdout.splitlines() if l.startswith("BIAS="))
    values = json.loads(marker.removeprefix("BIAS="))
    # every term is active on these coordinates, and they add up
    assert min(values["parts"]) > 1.0e-6
    assert abs(values["energy"] - sum(values["parts"])) < 1.0e-14
    assert values["fd"] < 1.0e-7
    # the massless site belongs to no restraint group and gets no force
    assert values["massless_force"] == 0.0
    assert values["settings"]["odp"]["window"] == 3
    assert values["settings"]["droplet"]["enabled"] is True


DECK = """\
hf/sto-3g
md(nstep=3,dt=0.5,ensemble=nvt,temperature=300,friction=5.0,seed=7,trajectory_interval=1,restart_interval=3)
qmmm(pdb_file="formaldehyde_water.pdb",forcefield_files="formaldehyde.xml tip3p.xml",qm_atoms="0-3")
droplet(enabled=true,radius=3.0,buffer=0.5,force_constant=2.0,max_penetration=5.0)
solute_com(enabled=true,force_constant=1.0,atoms="0-3")
odp(enabled=true,cv="distance(1,5); distance(2,5)",scale="1.0,1.0",reference_r="7.6,5.3",reference_p="5.3,6.4",center=0.2,k_parallel=0.1,k_perpendicular=0.02,window=2)
geom="../geometries/CH2O-2bc62dda4b8a.xyz"
"""

READ_SCRIPT = r"""
import json, sys, warnings
import numpy as np
from oqp.library.odp import odp_wham, read_odp_series
table = sys.argv[1]
data = np.load(table)
series = read_odp_series(table)
kj = 2625.499639
with warnings.catch_warnings():
    warnings.simplefilter("error")
    wham = odp_wham([table], 300.0, bins=3)
split = data["E_pot"] - data["E_unbiased"] - kj*(
    data["odp_bias_hartree"] + data["droplet_energy_hartree"]
    + data["solute_com_energy_hartree"])
try:
    odp_wham([table], 310.0, bins=3)
    rejected = None
except ValueError as exc:
    rejected = str(exc)
print("READ=" + json.dumps({
    "ensemble": series["ensemble"],
    "temperature": series["thermostat_temperature_kelvin"],
    "window": int(series["provenance"]["window"]),
    "n": int(len(series["xi"])),
    "steps": [int(v) for v in series["step"]],
    "xi": [float(v) for v in series["xi"]],
    "split": float(np.max(np.abs(split))),
    "droplet": float(np.max(data["droplet_energy_hartree"])),
    "com": float(np.max(data["solute_com_energy_hartree"])),
    "warning": wham["ensemble_warning"],
    "rejected": rejected,
}))
"""


def test_qmmm_md_odp_window_runs_and_feeds_wham(tmp_path):
    pytest.importorskip("openmm")
    work = tmp_path / "QMMM"
    work.mkdir()
    (tmp_path / "geometries").mkdir()
    for name in ("formaldehyde_water.pdb", "formaldehyde.xml", "tip3p.xml"):
        shutil.copy(EXAMPLES / name, work / name)
    shutil.copy(ROOT / "examples" / "geometries" / "CH2O-2bc62dda4b8a.xyz",
                tmp_path / "geometries")
    (work / "window.oqp").write_text(DECK)
    env = child_env()
    env["OMP_NUM_THREADS"] = "2"
    run = subprocess.run(
        [sys.executable, "-m", "oqp.pyoqp", "--silent", "--nompi", "window.oqp"],
        cwd=work, env=env, capture_output=True, text=True, check=False)
    if run.returncode != 0:
        if "cannot load" in run.stderr or "No module named" in run.stderr:
            pytest.skip("compiled OpenQP runtime is not available")
        pytest.fail(run.stdout[-4000:] + run.stderr[-4000:])
    tables = sorted(work.glob("*.npz"))
    table = next(p for p in tables if "restart" not in p.name)
    read = subprocess.run([sys.executable, "-c", READ_SCRIPT, str(table)],
                          cwd=work, env=child_env(), capture_output=True,
                          text=True, check=False)
    if read.returncode != 0:
        pytest.fail(read.stdout + read.stderr)
    marker = next(l for l in read.stdout.splitlines() if l.startswith("READ="))
    values = json.loads(marker.removeprefix("READ="))
    assert values["ensemble"] == "NVT"
    assert values["temperature"] == 300.0
    assert values["window"] == 2
    assert values["n"] == 4
    # the propagated potential is the QM/MM energy plus every bias term
    assert values["split"] < 1.0e-6
    assert values["droplet"] > 0.0 and values["com"] > 0.0
    assert values["warning"] is None
    assert "sampled at 300 K" in values["rejected"]

    # keep the step-3 checkpoint, then continue the window to step 5: the
    # table keeps the earlier rows and the continued rows carry the same records
    checkpoints = sorted(p for p in work.glob("*restart*.npz"))
    assert checkpoints
    saved = {p: p.read_bytes() for p in checkpoints}
    (work / "window.oqp").write_text(DECK.replace(
        "md(nstep=3,", "md(nstep=5,restart=true,"))
    run = subprocess.run(
        [sys.executable, "-m", "oqp.pyoqp", "--silent", "--nompi", "window.oqp"],
        cwd=work, env=env, capture_output=True, text=True, check=False)
    if run.returncode != 0:
        pytest.fail(run.stdout[-4000:] + run.stderr[-4000:])
    read = subprocess.run([sys.executable, "-c", READ_SCRIPT, str(table)],
                          cwd=work, env=child_env(), capture_output=True,
                          text=True, check=False)
    if read.returncode != 0:
        pytest.fail(read.stdout + read.stderr)
    marker = next(l for l in read.stdout.splitlines() if l.startswith("READ="))
    continued = json.loads(marker.removeprefix("READ="))
    assert continued["n"] == 6
    assert continued["steps"] == [0, 1, 2, 3, 4, 5]
    assert continued["xi"][:4] == values["xi"]
    assert continued["split"] < 1.0e-6
    assert continued["window"] == 2

    # from the older step-3 checkpoint, a continuation with a different bias
    # is refused before any output is truncated to step 3: the trajectory,
    # log and table still hold steps 4 and 5
    for path, data in saved.items():
        path.write_bytes(data)
    outputs = {p: p.read_bytes() for p in work.iterdir()
               if p.is_file() and p not in saved
               and (p.name.startswith("qmmm_") or p.suffix == ".npz")}
    assert table in outputs and any(p.suffix == ".pdb" for p in outputs)
    (work / "window.oqp").write_text(DECK.replace(
        "md(nstep=3,", "md(nstep=5,restart=true,").replace(
        "center=0.2,", "center=0.3,"))
    run = subprocess.run(
        [sys.executable, "-m", "oqp.pyoqp", "--silent", "--nompi", "window.oqp"],
        cwd=work, env=env, capture_output=True, text=True, check=False)
    assert run.returncode != 0
    assert "differs from this run in controls" in run.stdout + run.stderr
    for path, data in outputs.items():
        assert path.read_bytes() == data, path.name


def test_bias_atom_selections_are_validated():
    from oqp.library.qmmm_md_bias import _index_list

    assert _index_list("0-3,7 9:11", 20, "[solute_com]") == [
        0, 1, 2, 3, 7, 9, 10, 11]
    for spec, message in (("-1", "not a 0-based index"),
                          ("3-1", "descends"),
                          ("0-20", "outside 0..19"),
                          ("a", "not a 0-based index")):
        with pytest.raises(ValueError, match=message):
            _index_list(spec, 20, "[solute_com]")


def test_wham_refuses_tampered_qmmm_md_records(tmp_path):
    from oqp.library.odp import read_qmmm_md_odp_series

    meta = {"format": "openqp-qmmm-md-odp-series-1", "ensemble": "NVT",
            "thermostat": {"temperature_kelvin": 300.0},
            "controls": {"odp": {
                "enabled": True, "window": 1, "cv": ["distance(1,2)"],
                "cv_atom_indexing": "1-based", "cv_native_units": ["bohr"],
                "scale": [1.0], "reference_r": [2.0], "reference_p": [3.0],
                "scaled_path_length": 1.0, "center": 0.5,
                "k_parallel_hartree": 0.1, "k_perpendicular_hartree": 0.0,
                "projection": "signed_scaled_dot_over_path_norm_squared",
                "perpendicular_restraint": False},
                "droplet": {"enabled": False}, "solute_com": {"enabled": False}},
            "system_identity": {"kind": "qmmm_md", "sha256": "x"}}
    raw = np.array([[2.2], [2.6]])
    xi = raw[:, 0] - 2.0
    arrays = dict(
        step=np.array([0, 1]), time_ps=np.array([0.0, 0.001]),
        E_tot=np.zeros(2), e_unbiased_pot_hartree=np.zeros(2),
        odp_window=np.array([1, 1]), odp_xi=xi, odp_cv_raw=raw,
        odp_cv_scaled=raw.copy(), odp_cv_perpendicular=np.zeros((2, 1)),
        odp_perpendicular_norm=np.zeros(2),
        odp_bias_parallel_hartree=0.05*(xi - 0.5)**2,
        odp_bias_perpendicular_hartree=np.zeros(2),
        odp_bias_hartree=0.05*(xi - 0.5)**2,
        bias_metadata_json=np.array([json.dumps(meta)]))
    good = tmp_path / "good.npz"
    np.savez(good, **arrays)
    series = read_qmmm_md_odp_series(good)
    assert np.array_equal(series["xi"], xi)
    arrays["odp_bias_hartree"] = arrays["odp_bias_hartree"] + 1.0e-6
    bad = tmp_path / "bad.npz"
    np.savez(bad, **arrays)
    with pytest.raises(ValueError, match="bias energy records disagree"):
        read_qmmm_md_odp_series(bad)


def test_window_identity_binds_the_mm_hamiltonian(tmp_path):
    pytest.importorskip("openmm")
    from oqp.library.qmmm_md_bias import mm_hamiltonian_identity

    options = {"frontier_scheme": "none", "ewald_tol": None, "lj_switch": False,
               "h_lj": False, "mm_charge_width": None}
    local = tmp_path / "solute.xml"
    shutil.copy(EXAMPLES / "formaldehyde.xml", local)
    base = mm_hamiltonian_identity([str(local), "tip5p.xml"], options)
    # a force field shipped with OpenMM is found and hashed by content
    assert base["forcefield_files"][1][0] == "tip5p.xml"
    assert len(base["forcefield_files"][1][1]) == 64
    assert mm_hamiltonian_identity([str(local), "tip5p.xml"], options) == base
    # the same file name with other contents is another Hamiltonian
    local.write_text(local.read_text().replace("</ForceField>",
                                               "<!-- edited -->\n</ForceField>"))
    assert mm_hamiltonian_identity([str(local), "tip5p.xml"], options) != base
    shutil.copy(EXAMPLES / "formaldehyde.xml", local)
    for key, value in (("ewald_tol", 1.0e-6), ("lj_switch", True),
                       ("h_lj", True), ("mm_charge_width", 0.7),
                       ("frontier_scheme", "rcd")):
        changed = dict(options, **{key: value})
        assert mm_hamiltonian_identity([str(local), "tip5p.xml"], changed) != base
    with pytest.raises(FileNotFoundError):
        mm_hamiltonian_identity(["no-such-forcefield.xml"], options)


def test_periodic_bias_refuses_a_triclinic_cell():
    from oqp.library.qmmm_md_bias import require_orthorhombic

    require_orthorhombic(np.diag([3.0, 3.1, 3.2]), ["odp"])
    triclinic = np.diag([3.0, 3.1, 3.2])
    triclinic[1, 0] = 0.5
    with pytest.raises(ValueError, match="requires an orthorhombic cell"):
        require_orthorhombic(triclinic, ["odp", "droplet"])
