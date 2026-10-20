"""FlexiBLE QM/MM boundary: plugin reference, derivatives, periodic images."""

import json
from pathlib import Path

import numpy as np
import pytest

from oqp.library.flexible_boundary import (
    KB_HARTREE_PER_KELVIN, FlexiBLEBoundary, pair_term,
)

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "tests" / "data" / "flexible_boundary_plugin_reference.json"
BOHR_PER_NM = 10.0/0.52917721092


def _water_molecules(nmol):
    return [[3*i, 3*i + 1, 3*i + 2] for i in range(nmol)]


def _boundary_from_case(case):
    nmol = len(case["positions_nm"])//3
    qm = set(case["qm_molecules"])
    molecules = _water_molecules(nmol)
    center = (("point", np.asarray(case["center_nm"])*BOHR_PER_NM)
              if case["boundary"] == "point" else ("molecules", None))
    return FlexiBLEBoundary(
        [m for i, m in enumerate(molecules) if i in qm],
        [m for i, m in enumerate(molecules) if i not in qm],
        np.asarray(case["masses_dalton"]),
        alpha=case["alpha_per_nm"]/BOHR_PER_NM,
        temperature=case["temperature_kelvin"],
        anchor=None if case["anchor"] < 0 else case["anchor"],
        center=center, threshold=case["threshold"], scale=case["scale"],
        max_iterations=case["max_iterations"],
        keep_below_threshold=case["keep_below_threshold"])


def test_matches_authors_openmm_plugin():
    data = json.loads(REFERENCE.read_text())
    gas_constant = data["provenance"]["plugin_gas_constant_kj_mol_k"]
    seen_terms = []
    for name, case in data["cases"].items():
        boundary = _boundary_from_case(case)
        xyz = np.asarray(case["positions_nm"])*BOHR_PER_NM
        energy, force = boundary.evaluate(xyz)
        kt = KB_HARTREE_PER_KELVIN*case["temperature_kelvin"]
        rt = gas_constant*case["temperature_kelvin"]
        ln_ratio = -energy/kt
        ln_ratio_plugin = -case["energy_kj_mol"]/rt
        assert abs(ln_ratio - ln_ratio_plugin) < 1.0e-12, name
        reduced = force/kt*BOHR_PER_NM
        reduced_plugin = np.asarray(case["forces_kj_mol_nm"])/rt
        scale = np.max(np.abs(reduced_plugin))
        assert np.max(np.abs(reduced - reduced_plugin)) < 1.0e-12*scale, name
        seen_terms.append(boundary.last["denominator"])
    # every case has a partition sum with more than the ordered term
    assert min(seen_terms) > 1.5


def _cluster(seed, nmol=40, radius=16.0):
    rng = np.random.default_rng(seed)
    oxygens = []
    while len(oxygens) < nmol:
        o = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(o) < radius and all(
                np.linalg.norm(o - p) > 4.8 for p in oxygens):
            oxygens.append(o)
    xyz = []
    for o in oxygens:
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        v = rng.normal(size=3)
        v -= v.dot(u)*u
        v /= np.linalg.norm(v)
        xyz += [o, o + 1.81*u, o + 1.81*(np.cos(1.824)*u + np.sin(1.824)*v)]
    # a non-symmetric three-atom solute near the origin
    xyz += [[0.31, -0.22, 0.17], [2.9, 0.4, -0.6], [-0.7, 2.6, 1.1]]
    return np.asarray(xyz)


def _solute_boundary(xyz, nqm, swaps, anchor=0, **kwargs):
    nmol = (len(xyz) - 3)//3
    solute = [3*nmol, 3*nmol + 1, 3*nmol + 2]
    masses = np.concatenate([np.tile([15.9994, 1.00794, 1.00794], nmol),
                             [12.0, 14.0067, 15.9994]])
    center = masses[solute] @ xyz[solute]/masses[solute].sum()
    order = list(np.argsort(np.linalg.norm(xyz[0:3*nmol:3] - center, axis=1)))
    qm = order[:nqm]
    for k in range(swaps):
        qm[nqm - 1 - k] = order[nqm + k]
    molecules = _water_molecules(nmol)
    options = dict(alpha=2.65, temperature=300.0, anchor=anchor,
                   center=("atoms", solute))
    options.update(kwargs)
    return FlexiBLEBoundary(
        [molecules[i] for i in sorted(qm)],
        [m for i, m in enumerate(molecules) if i not in qm], masses, **options)


@pytest.mark.parametrize("anchor", [0, None])
def test_force_is_the_exact_derivative_with_a_moving_solute_center(anchor):
    xyz = _cluster(11)
    boundary = _solute_boundary(xyz, nqm=6, swaps=2, anchor=anchor)
    energy, force = boundary.evaluate(xyz)
    assert energy > 0.0 and boundary.last["numerator"] < 0.5
    h = 1.0e-6
    rng = np.random.default_rng(5)
    atoms = list(rng.choice(len(xyz) - 3, size=10, replace=False)) + [
        len(xyz) - 3, len(xyz) - 2, len(xyz) - 1]
    worst = 0.0
    for atom in atoms:
        for axis in range(3):
            plus = xyz.copy()
            plus[atom, axis] += h
            minus = xyz.copy()
            minus[atom, axis] -= h
            fd = -(boundary.evaluate(plus)[0]
                   - boundary.evaluate(minus)[0])/(2.0*h)
            worst = max(worst, abs(fd - force[atom, axis]))
    assert worst < 1.0e-9
    # the potential depends only on distances to the solute centre
    assert np.linalg.norm(force.sum(axis=0)) < 1.0e-14
    torque = np.cross(xyz, force).sum(axis=0)
    assert np.linalg.norm(torque) < 1.0e-12


def test_pair_term_and_its_derivative():
    alpha = 0.7
    r = np.array([-1.0, 0.0, 0.3, 1.7, 4.0])
    value, derivative = pair_term(alpha, r)
    ar = alpha*np.maximum(r, 0.0)
    assert np.array_equal(value[:2], [0.0, 0.0])
    assert np.max(np.abs(value - ar**3/(1.0 + ar))) < 1.0e-15
    h = 1.0e-7
    fd = (pair_term(alpha, r + h)[0] - pair_term(alpha, r - h)[0])/(2.0*h)
    assert np.max(np.abs(fd[2:] - derivative[2:])) < 1.0e-7


def test_ordered_partition_reaches_no_other_term_beyond_threshold():
    xyz = _cluster(13)
    boundary = _solute_boundary(xyz, nqm=5, swaps=0)
    energy, _force = boundary.evaluate(xyz)
    assert boundary.last["numerator"] == 1.0
    assert boundary.last["denominator"] >= 1.0
    expected = KB_HARTREE_PER_KELVIN*300.0*np.log(boundary.last["denominator"])
    assert abs(energy - expected) <= 1.0e-14*abs(expected)


def test_periodic_images_are_irrelevant_and_ambiguity_is_refused():
    xyz = _cluster(17)
    box = np.array([47.0, 51.0, 53.0])
    boundary = _solute_boundary(xyz, nqm=6, swaps=2)
    reference = boundary.evaluate(xyz)
    rng = np.random.default_rng(3)
    shifted = xyz.copy()
    for start in range(0, len(xyz), 3):
        shifted[start:start + 3] += box*rng.integers(-2, 3, size=3)
    # move single atoms of the solute and of waters across faces as well
    shifted[-2] += box*np.array([1, 0, -1])
    shifted[4] -= box*np.array([0, 1, 0])
    boundary_box = _solute_boundary(xyz, nqm=6, swaps=2)
    periodic = boundary_box.evaluate(shifted, box=box)
    assert abs(periodic[0] - reference[0]) < 1.0e-14
    assert np.max(np.abs(periodic[1] - reference[1])) < 1.0e-13

    h = 1.0e-6
    for atom, axis in ((len(xyz) - 2, 0), (len(xyz) - 1, 2), (0, 1), (4, 2)):
        plus = shifted.copy()
        plus[atom, axis] += h
        minus = shifted.copy()
        minus[atom, axis] -= h
        fd = -(boundary_box.evaluate(plus, box=box)[0]
               - boundary_box.evaluate(minus, box=box)[0])/(2.0*h)
        assert abs(fd - periodic[1][atom, axis]) < 1.0e-9
    boundary_box.evaluate(shifted, box=box)

    # MM molecules beyond half the box are allowed: they meet QM molecules
    # only at smaller distances, where g = 0
    r = boundary_box.last["distances"]
    nqm = boundary_box.nqm
    cut = 0.5*(np.max(r[:nqm]) + np.max(r)) * 2.0
    assert np.max(r[:nqm]) < 0.5*cut < np.max(r)
    beyond = np.array([cut, 51.0, 53.0])
    clipped = boundary_box.evaluate(xyz, box=beyond)
    assert abs(clipped[0] - reference[0]) < 1.0e-14
    # a QM molecule beyond half the box is refused
    small = np.array([1.9*np.max(r[:nqm]), 51.0, 53.0])
    with pytest.raises(RuntimeError, match="QM molecule is not closer"):
        boundary_box.evaluate(xyz, box=small)
    whole = FlexiBLEBoundary(
        [[0, 1, 2]], [[3, 4, 5]], np.tile([16.0, 1.0, 1.0], 2),
        alpha=1.0, temperature=300.0, center=("molecules", None))
    with pytest.raises(ValueError, match="not defined in a periodic cell"):
        whole.evaluate(xyz[:6], box=box)


@pytest.mark.parametrize("alpha, tolerance", [(2.65, 1.0e-5), (1.0, 1.0e-3)])
def test_collected_partition_sum_approaches_the_exhaustive_sum(alpha, tolerance):
    """The reference kernel's truncated sum against all C(14, 5) partitions."""
    import itertools
    import math

    for seed in (11, 17, 23):
        xyz = _cluster(seed, nmol=14)
        boundary = _solute_boundary(xyz, nqm=5, swaps=1, alpha=alpha)
        boundary.evaluate(xyz)
        r = boundary.last["distances"]
        exhaustive = 0.0
        for qm in itertools.combinations(range(len(r)), 5):
            mm = [i for i in range(len(r)) if i not in qm]
            g, _ = pair_term(alpha, r[list(qm)][:, None] - r[mm][None, :])
            exhaustive += math.exp(-float(np.sum(g)))
        collected = boundary.last["denominator"]
        assert collected <= exhaustive*(1.0 + 1.0e-14)
        assert math.log(exhaustive/collected) < tolerance


def test_partition_explosion_is_refused_not_hidden():
    xyz = _cluster(19)
    boundary = _solute_boundary(xyz, nqm=8, swaps=1, alpha=0.05,
                                max_partitions=500)
    with pytest.raises(RuntimeError, match="alpha is too small"):
        boundary.evaluate(xyz)


def test_invalid_settings_and_hopeless_partition_are_rejected():
    masses = np.ones(6)
    with pytest.raises(ValueError, match="at least one QM and one MM"):
        FlexiBLEBoundary([], [[0, 1, 2]], masses, alpha=1.0, temperature=300.0)
    with pytest.raises(ValueError, match="alpha"):
        FlexiBLEBoundary([[0, 1, 2]], [[3, 4, 5]], masses, alpha=0.0,
                         temperature=300.0)
    with pytest.raises(ValueError, match="anchor"):
        FlexiBLEBoundary([[0, 1, 2]], [[3, 4, 5]], masses, alpha=1.0,
                         temperature=300.0, anchor=3)
    # the QM molecule far outside the MM one: h(actual) underflows
    boundary = FlexiBLEBoundary(
        [[0, 1, 2]], [[3, 4, 5]], masses, alpha=5.0, temperature=300.0,
        center=("point", np.zeros(3)))
    xyz = np.zeros((6, 3))
    xyz[0:3] = [20.0, 0.0, 0.0]
    xyz[3:6] = [1.0, 0.0, 0.0]
    with pytest.raises(RuntimeError, match="too far from the distance ordering"):
        boundary.evaluate(xyz)


def _flexible_deck(geometry, nstep):
    example = ROOT / "examples" / "QMMM"
    return "\n".join((
        "mrsf(nstate=2)/bhhlyp/6-31g*",
        ("namd(T0,scheme=Overlap,nstep=%d,dt=0.5,substep=50,init_temp=300,"
         "velocity=maxwell,seed=8128,rng_stream=7,first_hop_step=1,"
         "nacme_check=off,nacme_policy=off,nve_policy=warn,ensemble=nve,"
         "thermostat=off,trajectory_interval=1,restart_interval=1)" % nstep),
        "guess(type=huckel)",
        (f'qmmm(pdb_file="{example / "formaldehyde_water.pdb"}",'
         f'forcefield_files="{example / "formaldehyde.xml"} '
         f'{example / "tip3p.xml"}",qm_atoms="0-6",cutoff=NoCutoff,'
         'embedding=electrostatic)'),
        ('flexible(enabled=true,alpha=5.0,temperature=300,center=solute,'
         'atoms="0-3",anchor=O)'),
        f'geom="{geometry}"',
        "",
    ))


def _run_deck(path, workdir):
    import os
    import subprocess
    import sys

    from oqp_module_isolation import child_env
    env = child_env(os.environ.copy())
    env["OMP_NUM_THREADS"] = "4"
    return subprocess.run(
        [sys.executable, "-m", "oqp.pyoqp", "--silent", "--nompi", str(path)],
        cwd=workdir, env=env, capture_output=True, text=True, check=False)


def _read_trajectory(path, frames=False):
    import subprocess
    import sys

    from oqp_module_isolation import child_env
    script = (
        "import json,sys\n"
        "import numpy as np\n"
        "from oqp.library.namd import read_namd_trajectory\n"
        "h,r=read_namd_trajectory(sys.argv[1])\n"
        "print(json.dumps({'flexible':h['independent_controls'].get('flexible'),"
        "'steps':r['step'].tolist(),"
        "'restraint':r['conservative_restraint_energy_hartree'].tolist(),"
        "'etot':r['e_tot_hartree'].tolist(),"
        "'xyz':r['coordinates_bohr'][-1].tolist(),"
        "'frames':(r['coordinates_bohr'].tolist() if len(sys.argv) > 2 "
        "else None)}))\n")
    arguments = [sys.executable, "-c", script, str(path)] + (
        ["frames"] if frames else [])
    result = subprocess.run(arguments,
                            env=child_env(), capture_output=True, text=True,
                            check=False)
    if result.returncode != 0:
        pytest.fail(result.stdout + result.stderr)
    return json.loads(result.stdout.splitlines()[-1])


def test_namd_qmmm_flexible_boundary_conserves_energy_and_restarts(tmp_path):
    pytest.importorskip("openmm")
    import re

    # QM: formaldehyde and the water above its oxygen; MM: four waters
    pdb = (ROOT / "examples" / "QMMM" / "formaldehyde_water.pdb").read_text()
    atoms = [line for line in pdb.splitlines() if line.startswith("HETATM")][:7]
    geometry = tmp_path / "qm.xyz"
    geometry.write_text("7\nformaldehyde + water\n" + "\n".join(
        f"{line[76:78].strip()} {line[30:38]} {line[38:46]} {line[46:54]}"
        for line in atoms) + "\n")

    straight = tmp_path / "straight"
    straight.mkdir()
    deck = straight / "flex.oqp"
    deck.write_text(_flexible_deck(geometry, 3))
    result = _run_deck(deck, straight)
    if result.returncode != 0:
        if "cannot load" in result.stderr or "No module named" in result.stderr:
            pytest.skip("compiled OpenQP runtime is not available")
        pytest.fail(result.stdout[-4000:] + result.stderr[-4000:])
    values = _read_trajectory(straight / "flex.namd.trj")
    assert values["flexible"]["qm_molecules"] == 1
    assert values["flexible"]["mm_molecules"] == 4
    assert values["steps"] == [0, 1, 2, 3]
    assert min(values["restraint"]) > 1.0e-5

    # the same trajectory without the boundary: the NVE energy change per
    # step must agree, so the FlexiBLE force is the gradient of its energy
    plain = tmp_path / "plain"
    plain.mkdir()
    deck = plain / "flex.oqp"
    deck.write_text(_flexible_deck(geometry, 3).replace(
        "flexible(enabled=true", "flexible(enabled=false"))
    result = _run_deck(deck, plain)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    reference = _read_trajectory(plain / "flex.namd.trj")
    assert reference["flexible"] is None
    assert max(reference["restraint"]) == 0.0
    drift = np.asarray(values["etot"]) - values["etot"][0]
    drift_plain = np.asarray(reference["etot"]) - reference["etot"][0]
    assert np.max(np.abs(drift - drift_plain)) < 1.0e-6

    resumed = tmp_path / "resumed"
    resumed.mkdir()
    deck = resumed / "flex.oqp"
    deck.write_text(_flexible_deck(geometry, 2))
    result = _run_deck(deck, resumed)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    manifest = resumed / "flex.namd.restart.oqp"
    text = manifest.read_text()
    extended = re.sub(r"\bnstep=2\b", "nstep=3", text, count=1)
    assert extended != text and "restart=true" in extended
    manifest.write_text(extended)
    result = _run_deck(manifest, resumed)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    with np.load(resumed / "flex.namd.restart.npz", allow_pickle=False) as saved:
        assert int(saved["step"][0]) == 3
        flexible_energy = float(saved["flexible_energy"][0])
    continued = _read_trajectory(resumed / "flex.namd.trj")
    assert continued["steps"] == [0, 1, 2, 3]
    assert abs(flexible_energy - values["restraint"][-1]) < 1.0e-9
    assert np.max(np.abs(np.asarray(continued["xyz"])
                         - np.asarray(values["xyz"]))) < 1.0e-8


# QM: the C-terminal amide of alanine dipeptide and water HOH 76, the water
# nearest to it; MM: the other 105 waters of the periodic box
PME_FLEXIBLE_DECK = """\
mrsf(nstate=2)/bhhlyp/6-31g namd(S0,scheme=Overlap,nstep=3,seed=11,restart_interval=3)
scf(conv=1e-09)
qmmm(forcefield_files="amber14-all.xml amber14/tip3p.xml",qm_atoms="8,9,16,17,18,244,245,246",cutoff=PME,ewald_tol=1e-06,lj_switch=true,h_lj=true,mm_charge_width=0.7)
flexible(enabled=true,alpha=5.0,temperature=300,atoms="8,9,16,17,18")
geom="ala_box.pdb"
"""


def test_periodic_pme_namd_qmmm_flexible_boundary_uses_the_cell(tmp_path):
    """Periodic NAMD QM/MM: the recorded FlexiBLE energy is the minimum-image
    potential of the recorded frames.

    Energy conservation is not compared here: this periodic ESPF example
    drifts by about 3e-4 Hartree per step without any boundary, so a
    with/without comparison measures trajectory divergence, not FlexiBLE.
    The periodic force is checked against finite differences in
    test_periodic_images_are_irrelevant_and_ambiguity_is_refused.
    """
    openmm_app = pytest.importorskip("openmm.app")
    import shutil

    example = ROOT / "examples" / "QMMM"
    workdir = tmp_path / "pme"
    workdir.mkdir()
    shutil.copy(example / "ala_box.pdb", workdir / "ala_box.pdb")
    deck = workdir / "pme.oqp"
    deck.write_text(PME_FLEXIBLE_DECK)
    result = _run_deck(deck, workdir)
    if result.returncode != 0:
        if "cannot load" in result.stderr or "No module named" in result.stderr:
            pytest.skip("compiled OpenQP runtime is not available")
        pytest.fail(result.stdout[-4000:] + result.stderr[-4000:])
    values = _read_trajectory(workdir / "pme.namd.trj", frames=True)
    assert values["flexible"]["qm_molecules"] == 1
    assert values["flexible"]["mm_molecules"] == 105
    assert min(values["restraint"]) > 1.0e-6

    pdb = openmm_app.PDBFile(str(example / "ala_box.pdb"))
    qm_water = [244, 245, 246]
    waters = [[a.index for a in residue.atoms()]
              for residue in pdb.topology.residues() if residue.name == "HOH"]
    # the masses NAMD integrates with: the force field's, not the elements'
    system = openmm_app.ForceField(
        "amber14-all.xml", "amber14/tip3p.xml").createSystem(pdb.topology)
    masses = np.array([system.getParticleMass(i)._value
                       for i in range(system.getNumParticles())])
    boundary = FlexiBLEBoundary(
        [qm_water], [w for w in waters if w != qm_water], masses,
        alpha=5.0*0.52917721092, temperature=300.0, anchor=0,
        center=("atoms", [8, 9, 16, 17, 18]))
    box = np.full(3, 16.0/0.52917721092)
    for xyz, recorded in zip(values["frames"], values["restraint"]):
        energy, _force = boundary.evaluate(np.asarray(xyz), box=box)
        assert abs(energy - recorded) < 1.0e-12


def test_flexible_boundary_and_odp_window_example_runs_together(tmp_path):
    """examples/QMMM/ala-box_BHHLYP-MRSF-NAMD-QMMM-PME-FLEXIBLE-ODP.oqp"""
    pytest.importorskip("openmm")
    import shutil
    import subprocess
    import sys

    from oqp_module_isolation import child_env
    name = "ala-box_BHHLYP-MRSF-NAMD-QMMM-PME-FLEXIBLE-ODP"
    example = ROOT / "examples" / "QMMM"
    shutil.copy(example / f"{name}.oqp", tmp_path)
    shutil.copy(example / "ala_box.pdb", tmp_path)
    result = _run_deck(tmp_path / f"{name}.oqp", tmp_path)
    if result.returncode != 0:
        if "cannot load" in result.stderr or "No module named" in result.stderr:
            pytest.skip("compiled OpenQP runtime is not available")
        pytest.fail(result.stdout[-4000:] + result.stderr[-4000:])
    script = (
        "import json,sys,warnings\n"
        "import numpy as np\n"
        "from oqp.library.namd import read_namd_trajectory\n"
        "from oqp.library.odp import odp_wham\n"
        "h,r=read_namd_trajectory(sys.argv[1])\n"
        "with warnings.catch_warnings():\n"
        "    warnings.simplefilter('error')\n"
        "    w=odp_wham([sys.argv[1]],300.0,bins=3)\n"
        "print(json.dumps({'ensemble':h['ensemble'],"
        "'flexible':h['independent_controls']['flexible'],"
        "'window':h['odp']['window'],'steps':r['step'].tolist(),"
        "'restraint':r['conservative_restraint_energy_hartree'].tolist(),"
        "'bias':r['odp_bias_hartree'].tolist(),"
        "'wham_warning':w['ensemble_warning']}))\n")
    inspection = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / f"{name}.namd.trj")],
        env=child_env(), capture_output=True, text=True, check=False)
    if inspection.returncode != 0:
        pytest.fail(inspection.stdout + inspection.stderr)
    values = json.loads(inspection.stdout.splitlines()[-1])
    assert values["ensemble"] == "NVT"
    assert values["flexible"]["qm_molecules"] == 1
    assert values["flexible"]["mm_molecules"] == 105
    assert values["window"] == 1
    assert values["steps"] == [0, 1, 2]
    assert min(values["restraint"]) > 1.0e-6
    assert min(values["bias"]) > 1.0e-6
    assert values["wham_warning"] is None
