"""
pose_restraints.py

Scaffold for pose-acting restraint terms to extend the boltz_ext restraint
machinery (Ishitani & Moriwaki, ACS Omega 2025).

DESIGN INTENT
-------------
The published restraints (chiral volume, bond length, bond angle) act on LOCAL
intramolecular geometry and are provably orthogonal to ligand pose RMSD
(Table 1: Bond RMSD -> 0.0003 A while ligand RMSD is unchanged). To affect
POSE you must restrain degrees of freedom that actually control placement:
torsions and protein-ligand contacts.

This module provides pose-acting energy terms designed to plug into the
existing Restraints machinery in restraints.py the SAME WAY chiral/bond/angle
restraints do: one small data object per restrained quantity, registered via
Restraints.register_site() against runtime atom slots, with setup()/is_valid()/
calc()/grad() methods that Restraints.calc()/.grad() call in their per-batch
loops.

  1. PoseTorsionData : penalizes a single ligand torsion for falling outside
                       the range spanned by an RDKit conformer ENSEMBLE (a
                       prior, not a single reference value). One instance is
                       created per rotatable bond by
                       Restraints.make_pose_torsion_restraints().

  2. clash_energy / PoseClashData (optional, protein-ligand steric term) is
     sketched at the bottom of this file as a follow-up extension; not wired
     into restraints.py by default since it needs protein coordinates that
     are not yet threaded through schema.py's parsing call site.

ATOM INDEX HANDLING
--------------------
This module builds torsion quads directly on the RDKit mol as passed in
(e.g. ref_mol from schema.py), i.e. RDKit's OWN atom numbering, NOT the
`atoms`-list / active_sites numbering used elsewhere in restraints.py.

Translating RDKit indices -> atoms-list positions is the CALLER's job, exactly
mirroring how make_bond() in schema.py already handles this:

    idx_1 = idx_map[idx_1]
    idx_2 = idx_map[idx_2]

Restraints.make_pose_torsion_restraints() (in restraints.py) does this
translation via idx_map, and silently skips any quad touching an atom not
present in idx_map (e.g. filtered/absent atoms) -- exactly like the existing
bond-restraint loop in schema.py does. Do NOT strip or add hydrogens on the
mol passed into build_torsion_prior(); leave it exactly as the caller's
ref_mol, so quad indices line up with idx_map's keys.

IMPORTANT CAVEATS
-----------------
* This is a SCAFFOLD. It is not guaranteed to improve pose RMSD; that is the
  empirical question the dissertation tests. Treat w_torsion and start_sigma
  as hyperparameters to sweep, with unmodified Boltz R as baseline.
* Pose-acting restraints are sensitive to WHEN they fire during reverse
  diffusion. Too late (very low sigma) and the ligand is already placed;
  too early and the term fights the denoiser. Sweep a separate
  pose_start_sigma from the geometry restraints' start_sigma.
* Always co-report chirality/bond/angle metrics alongside pose RMSD, to show
  the pose term does not regress what the existing restraints already fixed.

Author: (your name) -- MSc Bioinformatics dissertation scaffold
"""

from __future__ import annotations
import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem import rdMolTransforms
    _HAVE_RDKIT = True
except ImportError:  # allow import on cluster nodes without RDKit in the path
    _HAVE_RDKIT = False


# ----------------------------------------------------------------------------
# Torsion prior from an RDKit conformer ensemble
# ----------------------------------------------------------------------------

def _enumerate_rotatable_torsions(mol):
    """Return list of (i, j, k, l) atom-index quadruples for rotatable bonds.

    Indices are RDKit's own atom numbering for `mol` (the same mol passed
    into build_torsion_prior) -- NOT translated through any idx_map. The
    caller (Restraints.make_pose_torsion_restraints) is responsible for that
    translation.

    Uses RDKit's rotatable-bond SMARTS, then picks a heavy-atom neighbour on
    each side to define the dihedral. Terminal/degenerate cases are skipped.
    """
    rot_smarts = Chem.MolFromSmarts("[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]")
    quads = []
    for (a, b) in mol.GetSubstructMatches(rot_smarts):
        atom_a = mol.GetAtomWithIdx(a)
        atom_b = mol.GetAtomWithIdx(b)
        nbr_a = [n.GetIdx() for n in atom_a.GetNeighbors()
                 if n.GetIdx() != b and n.GetAtomicNum() > 1]
        nbr_b = [n.GetIdx() for n in atom_b.GetNeighbors()
                 if n.GetIdx() != a and n.GetAtomicNum() > 1]
        if nbr_a and nbr_b:
            quads.append((nbr_a[0], a, b, nbr_b[0]))
    return quads


def build_torsion_prior(ligand_rdkit_mol, n_conformers=50, random_seed=0xC0FFEE):
    """Generate an ensemble of low-energy conformers and record, for each
    rotatable torsion, the set of dihedral values it adopts.

    ligand_rdkit_mol is used exactly as passed -- this function does not
    add or strip hydrogens, so returned quad indices remain in the caller's
    own RDKit index space (see module docstring on ATOM INDEX HANDLING).

    Returns
    -------
    quads : list[tuple[int,int,int,int]]
        Atom-index quadruples defining each torsion, in ligand_rdkit_mol's
        own RDKit numbering.
    allowed : list[np.ndarray]
        For each torsion, an array of dihedral angles (radians) observed
        across the ensemble. This is the PRIOR: the term penalizes deviation
        from the NEAREST allowed value, so multi-modal rotamers are respected.
    """
    if not _HAVE_RDKIT:
        raise RuntimeError("RDKit required to build the torsion prior.")

    # Work on a copy so we never mutate the caller's mol in place.
    mol = Chem.Mol(ligand_rdkit_mol)
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = random_seed
    params.numThreads = 0
    cids = AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, params=params)

    # MMFF minimize so the ensemble represents low-energy conformers.
    # MMFF energies are less reliable without explicit hydrogens; if that
    # matters for your evaluation, minimize a separately-Hs-added copy for
    # embedding quality only, then transfer coordinates back before reading
    # dihedrals. Flagged here as a documented follow-up rather than silently
    # changing this function's index semantics.
    try:
        res = AllChem.MMFFOptimizeMoleculeConfs(
            mol, numThreads=0, maxIters=2000)
        n_bad = sum(1 for c, _ in res if c != 0)
        if n_bad:
            print('[pose_restraints] WARNING: %d/%d confs '
                  'not converged' % (n_bad, len(res)))
    except Exception:
        # MMFF may behave poorly without explicit Hs for some ligands;
        # embedding geometry alone is still a usable (if rougher) prior.
        print('[pose_restraints] WARNING: MMFF failed; '
              'using embedding geometry only')

    mol = Chem.RemoveHs(mol)

    quads = _enumerate_rotatable_torsions(mol)
    allowed = []
    for (i, j, k, l) in quads:
        vals = []
        for cid in cids:
            conf = mol.GetConformer(cid)
            ang = rdMolTransforms.GetDihedralRad(conf, i, j, k, l)
            vals.append(ang)
        allowed.append(np.asarray(vals, dtype=np.float64))
    return quads, allowed


def _dihedral(p0, p1, p2, p3):
    """Dihedral angle (radians) for four 3D points."""
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1 = b1 / (np.linalg.norm(b1) + 1e-12)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return np.arctan2(y, x)


# ----------------------------------------------------------------------------
# Runtime restraint object -- mirrors AngleData / ChiralData's contract
# ----------------------------------------------------------------------------

class PoseTorsionData:
    """Runtime torsion restraint for ONE rotatable bond.

    Mirrors the setup()/is_valid()/calc()/grad() contract that AngleData and
    ChiralData already implement, so it slots into
    Restraints.calc()/Restraints.grad()'s existing per-batch loops with a
    matching `for pd in self.pose_data: ...` block, and into setup_site()'s
    register_site() machinery the same way make_angle() wires up AngleData.

    One instance is created per rotatable-bond quad by
    Restraints.make_pose_torsion_restraints(); atom-slot indices (already
    translated through idx_map by the caller) are filled in later via
    setup(), when setup_site() replays the registered callbacks for this
    specific batch/structure.
    """

    def __init__(self, allowed_angles: np.ndarray, weight: float = 1.0):
        self.allowed = allowed_angles
        self.weight = float(weight)
        self.aid = [None, None, None, None]

    def setup(self, local_idx: int, slot: int) -> None:
        """Called via the register_site() callback for each of the 4 atoms
        in this torsion, once per slot (0=i, 1=j, 2=k, 3=l)."""
        self.aid[slot] = local_idx

    def is_valid(self) -> bool:
        return all(a is not None for a in self.aid)

    def calc(self, crds: np.ndarray) -> float:
        """crds: (natoms, 3) array for ONE structure in the batch, in the
        same local indexing as self.aid (i.e. active_sites-local indices)."""
        i, j, k, l = self.aid
        theta = _dihedral(crds[i], crds[j], crds[k], crds[l])
        d = np.angle(np.exp(1j * (theta - self.allowed)))  # wrapped diffs
        return self.weight * float(np.min(d * d))           # nearest rotamer

    def grad(self, crds: np.ndarray, grad: np.ndarray, h: float = 1e-4) -> None:
        """Central-difference gradient, accumulated in-place into `grad`
        (same shape as crds). Scoped to only this quad's 4 atoms -- cheap
        (24 calc() evaluations per quad per step), unlike differentiating
        the whole ligand at once.

        NOTE: mutates `crds` transiently (perturb, evaluate, restore) but
        leaves it unchanged on return.
        """
        i, j, k, l = self.aid
        for idx in (i, j, k, l):
            for dim in range(3):
                crds[idx, dim] += h
                ep = self.calc(crds)
                crds[idx, dim] -= 2 * h
                em = self.calc(crds)
                crds[idx, dim] += h  # restore
                grad[idx, dim] += (ep - em) / (2 * h)


# ----------------------------------------------------------------------------
# Optional follow-up: protein-ligand clash term (NOT wired into restraints.py
# yet -- needs protein coordinates threaded through schema.py's parse call
# site first). Left here so the extension point is obvious later.
# ----------------------------------------------------------------------------

def clash_energy(lig_coords, lig_radii,
                 prot_coords, prot_radii,
                 drel_threshold=0.75, neighbor_cutoff=6.0):
    """L_clash: soft penalty for protein-ligand steric overlap.

    Mirrors PoseBusters' relative distance d_rel(i,j) = d(i,j)/(r_i + r_j).
    Pairs with d_rel < drel_threshold contribute a quadratic penalty; pairs
    above threshold contribute nothing. Only pairs within neighbor_cutoff (A)
    are considered, for speed.

    lig_coords  : (Nl,3)   ligand heavy-atom coords (current step, local idx)
    lig_radii   : (Nl,)    ligand VdW radii
    prot_coords : (Np,3)   protein heavy-atom coords (fixed this step)
    prot_radii  : (Np,)    protein VdW radii
    """
    total = 0.0
    for li in range(lig_coords.shape[0]):
        d = np.linalg.norm(prot_coords - lig_coords[li], axis=1)
        near = np.where(d < neighbor_cutoff)[0]
        for pj in near:
            sumr = lig_radii[li] + prot_radii[pj]
            drel = d[pj] / (sumr + 1e-12)
            if drel < drel_threshold:
                total += (drel_threshold - drel) ** 2
    return total


# ----------------------------------------------------------------------------
# Self-test (runs only if RDKit is available). Exercises build_torsion_prior
# and PoseTorsionData directly -- does NOT touch restraints.py/schema.py,
# since those require the full Boltz runtime (feats, idx_map, etc.) that
# isn't available standalone.
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    if not _HAVE_RDKIT:
        print("RDKit not available; skipping self-test.")
    else:
        # Ibuprofen: has rotatable bonds, good smoke test for the torsion term.
        mol = Chem.MolFromSmiles("CC(C)Cc1ccc(cc1)C(C)C(=O)O")
        quads, allowed = build_torsion_prior(mol, n_conformers=20)
        print("n rotatable torsions:", len(quads))

        # Use one embedded conformer's coords as a stand-in for step coords,
        # and fake up local indices as identity (i.e. pretend idx_map is a
        # no-op) purely to exercise PoseTorsionData end to end.
        m = Chem.Mol(mol)
        AllChem.EmbedMolecule(m, AllChem.ETKDGv3())
        conf = m.GetConformer()
        coords = np.array([list(conf.GetAtomPosition(a))
                           for a in range(m.GetNumAtoms())])

        total_e = 0.0
        for (i, j, k, l), allow in zip(quads, allowed):
            pd = PoseTorsionData(allow, weight=1.0)
            pd.setup(i, 0)
            pd.setup(j, 1)
            pd.setup(k, 2)
            pd.setup(l, 3)
            e = pd.calc(coords)
            total_e += e
            grad = np.zeros_like(coords)
            pd.grad(coords, grad)
            print(f"quad=({i},{j},{k},{l}) energy={e:.4f} "
                  f"max|grad|={np.abs(grad).max():.4f}")
        print(f"total torsion energy (should be small for an ensemble "
              f"member): {total_e:.4f}")
