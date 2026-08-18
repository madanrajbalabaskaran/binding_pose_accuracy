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
| `scripts/` | New | Evaluation pipeline: system selection, structure extraction, config generation, alignment and aggregation |

## Usage

Restraint behaviour is controlled from the ligand block of the input YAML:

```yaml
pose_w_torsion: 1
pose_start_sigma: 1.0
```
Later PyTorch versions caused segmentation faults on import. Install in editable
mode (`pip install -e .`).

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





Run inference with:

```bash
boltz predict configs/eval_8sge.yaml --seed 42 --diffusion_samples 5 \
      --out_dir <output directory>
```

## Environment

The torsional module requires:
