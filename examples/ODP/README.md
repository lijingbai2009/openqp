# ODP umbrella sampling and WHAM

`H2CO_BHHLYP-MRSF-NAMD-ODP.inp` is a minimal native ODP umbrella window. For a
production PMF, copy it into several windows, assign a unique `window` and
`center` in `[odp]`, and give each run its own packed `trajectory_file`.

Analyze all windows with the public Python API example:

```console
python odp_wham.py --temperature 300 --output odp-wham.npz window-*.namd.trj
```

The supplied temperature is the canonical temperature assumed by WHAM. For a
free-energy window run each window canonically, `namd(...,ensemble=nvt,
thermostat=langevin)` as in `H2CO_BHHLYP-MRSF-NAMD-ODP-NVT.oqp`; WHAM then
requires the supplied temperature to equal the recorded thermostat
temperature. An NVE window is still accepted, but OpenQP records the ensemble
and warns that WHAM assumes canonical sampling.

ODP also runs in periodic QM/MM (`qmmm(cutoff=PME)`, orthorhombic cell). Each
CV is measured at its minimum image: all CV atoms are placed at the image
nearest the first CV atom, and the run stops if two CV atoms are then not
closer than half the shortest box length.

To exercise checkpoint continuation end to end, run:

```console
python odp_restart.py --workdir odp-restart-smoke
```

The example performs one native ODP step, extends the generated canonical
`restart=true` manifest to step two, runs that manifest, and verifies the final
checkpoint. Omit `--workdir` for an automatically removed temporary run.
