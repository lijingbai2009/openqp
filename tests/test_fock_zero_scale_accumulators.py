"""Zero-scaled J/K digestion in the replicated and low-memory atomic Fock paths.

int2_{rhf,urohf}_data_t_update and *_consume_shell skip the exchange updates
when scale_exchange == 0 (pure LDA/GGA/meta-GGA) and the Coulomb updates when
scale_coulomb == 0 (the short-range exchange pass of a range-separated
functional).  Both the per-thread replicated accumulator and the shared atomic
accumulator (OQP_FOCK_ATOMIC=1, the automatic low-memory mode) carry the
guard, so each functional class below is run in fresh processes with the
atomic accumulator off and on, serially and with four OpenMP threads.

* The run must enter the requested accumulator: the atomic run prints the
  "shared+atomic" banner and the replicated run does not.
* Every accumulator/thread combination must reproduce the serial replicated
  energy to 1e-9 Eh; serially, the atomic accumulator must also reproduce the
  final alpha/beta Fock matrices to 1e-9.
* The serial replicated energy is pinned to unmodified main (84f324bf8,
  before the guard), so a dropped or doubled J or K term fails.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

CHILD = r'''
import json, sys
import numpy as np
from oqp.pyoqp import Runner
system, scf, mult, functional = sys.argv[1:5]
config = {
    'input': {'system': system.replace(';', '\n'), 'method': 'hf',
              'basis': '6-31g*', 'runtype': 'energy', 'functional': functional},
    'guess': {'type': 'huckel', 'save_mol': 'False'},
    'scf': {'type': scf, 'multiplicity': mult, 'conv': '1.0e-9',
            'maxit': '100', 'save_molden': 'False'},
    'tests': {'exception': 'True'},
}
runner = Runner(project='fock', input_file=None, log='fock.log',
                input_dict=config, silent=1, usempi=False)
runner.run()
data = runner.mol.data
out = {'energy': float(runner.mol.energies[0]),
       'fock_a': np.asarray(data['OQP::FOCK_A'], dtype=float).ravel().tolist()}
if scf != 'rhf':
    out['fock_b'] = np.asarray(data['OQP::FOCK_B'], dtype=float).ravel().tolist()
print('RESULT ' + json.dumps(out))
'''

WATER = 'O 0 0 -0.0410616; H -0.5331943 0.5331943 -0.6144692; H 0.5331943 -0.5331943 -0.6144692'
OH = 'O 0 0 0; H 0 0.3 0.92'

# Serial replicated energies from unmodified main 84f324bf8 (no zero-scale
# guard), managed build on zeus (macOS x86_64, GCC 15, Accelerate ILP64).
CASES = {
    'rks-pbe': (WATER, 'rhf', '1', 'pbe', -76.32090487015014),
    'uks-pbe': (OH, 'uhf', '2', 'pbe', -75.63631967619115),
    'rks-cam-b3lyp': (WATER, 'rhf', '1', 'cam-b3lyp', -76.37952876256894),
    # Hybrid control: both J and K are digested (no guard taken).
    'rks-b3lyp': (WATER, 'rhf', '1', 'b3lypv1r', -76.40808725453417),
}


def _run(tmp_path, case, atomic, threads):
    system, scf, mult, functional, _ = CASES[case]
    work = tmp_path / f'{case}-a{atomic}-t{threads}'
    work.mkdir()
    env = {k: v for k, v in os.environ.items() if not k.startswith('OQP_')}
    env.update(OMP_NUM_THREADS=str(threads), OQP_FOCK_ATOMIC=str(atomic),
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               VECLIB_MAXIMUM_THREADS='1')
    proc = subprocess.run([sys.executable, '-c', CHILD, system, scf, mult, functional],
                          cwd=work, env=env, capture_output=True, text=True,
                          timeout=900)
    log = work / 'fock.log'
    text = proc.stdout + proc.stderr
    if log.exists():
        text += log.read_text(errors='replace')
    assert proc.returncode == 0, text[-4000:]
    banner = 'Fock accumulator: shared+atomic' in text
    assert banner == bool(atomic), f'{case} atomic={atomic} threads={threads}: wrong accumulator'
    line = [x for x in proc.stdout.splitlines() if x.startswith('RESULT ')][-1]
    return json.loads(line[len('RESULT '):])


@pytest.mark.parametrize('case', sorted(CASES))
def test_zero_scale_guard_matches_across_accumulators_and_threads(tmp_path, case):
    reference = _run(tmp_path, case, 0, 1)
    assert np.isfinite(reference['energy'])
    pinned = CASES[case][4]
    if pinned is not None:
        assert abs(reference['energy'] - pinned) <= 1e-8, (reference['energy'], pinned)
    # One thread: the atomic and replicated accumulators visit the quartets in
    # the same order but round differently (the atomic path stores each
    # product before the add, the replicated path may contract it into an
    # FMA); unmodified main differs by 1.5e-11 for UKS OH, so the final Fock
    # matrices are compared to 1e-9.  Four threads change the summation order
    # and the SCF path; the final Fock then differs at the SCF convergence
    # level (unmodified main: 3e-7 to 2e-6 for UKS OH at conv=1e-9), so only
    # the energy is compared there.
    for atomic, threads, fock_tol in [(1, 1, 1e-9), (0, 4, None), (1, 4, None)]:
        result = _run(tmp_path, case, atomic, threads)
        assert abs(result['energy'] - reference['energy']) <= 1e-9, (atomic, threads)
        for key in ('fock_a', 'fock_b'):
            if fock_tol is not None and key in reference:
                diff = np.max(np.abs(np.asarray(result[key]) - np.asarray(reference[key])))
                assert diff <= fock_tol, (key, atomic, threads, diff)
