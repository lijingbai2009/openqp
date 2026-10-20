"""A pFON convergence claim must use a density annealed to the final 1 K.

The temperature can drop from hundreds of kelvin to 1 K at the beginning of
an iteration.  Its energy and DIIS error still use the density built at the
previous temperature; accepting convergence on that iteration reports a hot
density as a 1 K solution.  The ROHF/PBE H2O example exposed this on Windows
MKL ILP64 (0.00007740 Eh away from its reference at four OpenMP threads).
"""

import os
import re
import subprocess
import sys

import pytest


DECK = """\
[input]
system=
 8   0.000000000   0.000000000  -0.041061554
 1  -0.533194329   0.533194329  -0.614469223
 1   0.533194329  -0.533194329  -0.614469223
charge=0
method=hf
basis=6-31g
runtype=energy
functional=pbe
d4=false

[guess]
type=huckel

[scf]
type=rohf
multiplicity=3
maxit=200
maxdiis=7
pfon=true
pfon_start_temp=2000.0
pfon_cooling_rate=50.0
pfon_nsmear=5.0
conv=1.0e-6

[dftgrid]
rad_npts=96
ang_npts=302
pruned=
"""


@pytest.mark.parametrize("threads", [1, 4])
def test_pfon_energy_uses_final_temperature_density(tmp_path, threads):
    inp = tmp_path / "h2o_rohf_pfon.inp"
    inp.write_text(DECK)
    env = dict(os.environ, OMP_NUM_THREADS=str(threads))
    proc = subprocess.run(
        [sys.executable, "-m", "oqp.pyoqp", str(inp)],
        cwd=tmp_path, env=env, text=True, capture_output=True, timeout=900)
    log = inp.with_suffix(".log").read_text() if inp.with_suffix(".log").exists() else ""
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    assert "SCF convergence achieved" in log, log[-3000:]
    # The printed temperature belongs to the *new* density formed after each
    # iteration's energy.  At least the preceding iteration must therefore
    # have formed a 1 K density before convergence is accepted.
    temperatures = [float(x) for x in re.findall(r"Temp:\s+([0-9.]+)", log)]
    assert len(temperatures) >= 2, log[-3000:]
    assert temperatures[-2:] == [1.0, 1.0], temperatures[-8:]
