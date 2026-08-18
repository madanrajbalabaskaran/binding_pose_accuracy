A torsional prior for restraint-guided protein–ligand pose prediction
This repository extends the restraint-guided inference implementation of Ishitani and Moriwaki (2025) with a torsional restraint derived from a per-ligand conformer ensemble, applied during reverse diffusion alongside the published chirality, bond-length and bond-angle restraints.

Accompanies the MSc dissertation A torsional prior for restraint-guided protein–ligand pose prediction in Boltz-1 (Newcastle University, 2026).
Provenance
Forked from cddlab/boltz_ext at commit 9d88b09, which is itself a fork of jwohlwend/boltz (Boltz-1). All inference uses the Boltz-1 checkpoint; no retraining or fine-tuning was performed.

Everything not listed in the next section is inherited from upstream and unmodified.
Contributions of this work
Path
Status
Description
src/boltz/model/modules/pose_restraints.py
New
Torsional prior: conformer-ensemble construction, dihedral energy, finite-difference gradient, and the PoseTorsionData runtime object
src/boltz/model/modules/restraints.py
Modified
Wires the torsional term into the existing per-batch minimisation loop
src/boltz/data/parse/schema.py
Modified
Parses the torsional restraint keys from the input YAML
configs/
New
Inference configurations for the 5bzl and 8sge evaluation systems, one per experimental condition

Method
For each ligand, a 50-conformer ensemble is generated with RDKit's ETKDGv3 embedding under a fixed random seed and minimised with the MMFF94 force field. The dihedral angle at each rotatable bond is recorded across the ensemble, giving a multi-modal set of observed torsion values per bond. During inference, the restraint penalises the squared circular deviation of each rotatable-bond dihedral from the nearest value in its allowed set, so any rotamer the ensemble actually visited incurs no penalty.

The term is evaluated only in the low-noise phase of reverse diffusion, controlled by the same noise-level threshold as the published restraints.
Usage
Restraint behaviour is set from the ligand block of the input YAML. Note that SMILES strings must be written as single-quoted YAML scalars: double-quoted scalars process backslash escape sequences and will silently corrupt stereochemistry-bearing SMILES.

Run inference with:

boltz predict configs/eval_8sge.yaml --seed 42 --diffusion_samples 5 \

      --out_dir <output directory>
Environment
python 3.11.5

torch==2.8.0+cu128

torch_cluster            # from the PyTorch Geometric wheel index

rdkit

Later PyTorch releases caused segmentation faults on importing the restraints module. Install in editable mode:

pip install -e .
Reproducibility note
pose_restraints.py contains the corrected conformer-ensemble construction described in the dissertation (explicit hydrogens added before embedding for correct MMFF94 atom typing, maxIters raised to 2000, hydrogens stripped before torsion enumeration so heavy-atom indices stay consistent with the caller). The uncorrected results reported alongside it were produced with an earlier revision of this file and cannot be reproduced from the current state of this repository.
Licence and attribution
Distributed under the terms of the upstream LICENSE. The restraint-guided inference framework is the work of Ishitani and Moriwaki; the underlying structure prediction model is Boltz-1 (Wohlwend et al., 2024).
References
Ishitani, R. and Moriwaki, Y. (2025) 'Improving stereochemical limitations in protein–ligand complex structure prediction', ACS Omega, 10(46), pp. 56075–56084. doi:10.1021/acsomega.5c07675.

Wohlwend, J. et al. (2024) 'Boltz-1: democratizing biomolecular interaction modeling', bioRxiv. doi:10.1101/2024.11.19.624167.

