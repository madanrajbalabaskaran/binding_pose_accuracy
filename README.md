# Ensemble-derived torsional prior for restraint-guided pose prediction

A fork of [boltz_ext](https://github.com/cddlab/boltz_ext) (Ishitani and
Moriwaki, 2025), itself a fork of [Boltz-1](https://github.com/jwohlwend/boltz),
adding a torsional restraint derived from a per-ligand conformer ensemble and
applied during reverse diffusion.

Branched from boltz_ext at commit `9d88b09`. Everything not listed below is
inherited from upstream and unmodified.

## Files added or modified in this work

| Path | Status | Purpose |
|---|---|---|
| `src/boltz/model/modules/pose_restraints.py` | New | Torsional prior: conformer-ensemble construction, dihedral energy and gradient, `PoseTorsionData` class |
| `src/boltz/model/modules/restraints.py` | Modified | Dispatches the torsional term alongside the published restraints; adds `pose_start_sigma` |
| `src/boltz/data/parse/schema.py` | Modified | Parses `pose_w_torsion` and `pose_start_sigma` from the input YAML |
| `configs/` | New | Inference configurations for the 5bzl and 8sge evaluation systems, one per condition |
| `analysis/` | New | Evaluation pipeline: system selection, structure extraction, config generation, alignment and aggregation |

## Method

For each ligand, a 50-conformer ensemble is generated with RDKit's ETKDGv3
embedding under a fixed random seed and minimised with the MMFF94 force field.
The dihedral angle at each rotatable bond is recorded across the ensemble,
giving a multi-modal set of observed torsion values per bond. During inference,
the restraint penalises the squared circular deviation of each rotatable-bond
dihedral from the nearest value in its allowed set, so any rotamer the ensemble
actually visited incurs no penalty. The term acts only in the low-noise phase of
reverse diffusion, controlled by a noise-level threshold.


## Usage

A complete input configuration looks like this:

```yaml
sequences:
  - protein:
      id: A
      sequence: "<PROTEIN SEQUENCE>"
      msa: <path to .a3m>
  - ligand:
      id: B
      smiles: '<SMILES>'
      chiral_restraints: true
      pose_restraints: true
      pose_w_torsion: 1.0
restraints_config:
  angle:
    weight: 1
  bond:
    weight: 1
  chiral:
    weight: 1
  start_sigma: 1.0
  gpu: false
  verbose: true
```
Two settings matter more than they appear to:

`gpu: false` is required. The torsional term is implemented only on the CPU
minimisation path, so setting this to `true` silently disables it while the
published chirality, bond and angle restraints continue to run.

SMILES strings must be written as single-quoted YAML scalars. Double-quoted
scalars process backslash escape sequences and will silently corrupt
stereochemistry-bearing SMILES.

An optional `pose_start_sigma` key, set in the ligand block, allows the
torsional term to fire at a different point in reverse diffusion from the
published restraints. It defaults to `start_sigma` and can be omitted, as in the
configurations used for the reported results.

Run inference with:

```bash
boltz predict configs/eval_8sge.yaml --seed 42 --diffusion_samples 5 \
      --out_dir <output directory>
```

## Environment

The torsional module requires:

```
python 3.11.5
torch==2.8.0+cu128
torch_cluster            # from the PyTorch Geometric wheel index
rdkit
```

Later PyTorch versions caused segmentation faults on import. Install in editable
mode:

```bash
pip install -e .
```
Paths in the analysis/ scripts are hard-coded to the PLINDER installation and
scratch directories used for this work, and must be edited before reuse. The
variables to change are defined at the top of each script.

## Notes

`pose_restraints.py` contains the corrected conformer-ensemble construction
(`maxIters=2000`, explicit hydrogens added before embedding and removed before
torsion enumeration). The uncorrected results reported in the accompanying
manuscript were produced with an earlier version of this file and cannot be
reproduced from this revision.

## Licence and attribution

Licensed under the terms of the upstream `LICENSE`. The restraint-guided
inference framework is the work of Ishitani and Moriwaki; the underlying model
is Boltz-1 (Wohlwend et al., 2024).

## References

Ishitani, R. and Moriwaki, Y. (2025) 'Improving stereochemical limitations in
protein–ligand complex structure prediction', *ACS Omega*, 10(46),
pp. 56075–56084. doi:10.1021/acsomega.5c07675.

Wohlwend, J. et al. (2024) 'Boltz-1: democratizing biomolecular interaction
modeling', *bioRxiv*. doi:10.1101/2024.11.19.624167.
