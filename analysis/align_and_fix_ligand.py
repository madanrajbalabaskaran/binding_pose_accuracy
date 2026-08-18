import sys, os
import numpy as np
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import biotite.structure.io.pdb as pdb
from rdkit import Chem
from rdkit.Chem import AllChem

def run(pred_cif, lig_chain, prot_chain, crystal_pdb, template_sdf, out_sdf):
    pred = pdbx.get_structure(pdbx.CIFFile.read(pred_cif), model=1)
    pred_prot = pred[(pred.chain_id == prot_chain) & struc.filter_amino_acids(pred)]
    pred_lig  = pred[pred.chain_id == lig_chain]
    if len(pred_lig) == 0:
        raise ValueError(f"no ligand atoms for chain {lig_chain}")

    xtal = pdb.get_structure(pdb.PDBFile.read(crystal_pdb), model=1)
    xtal_prot = xtal[struc.filter_amino_acids(xtal)]

    fitted, transform, fix_i, mob_i = struc.superimpose_homologs(xtal_prot, pred_prot)
    prot_rmsd = struc.rmsd(xtal_prot[fix_i], fitted[mob_i])
    print(f"  aligned on {len(fix_i)} atoms, protein RMSD = {prot_rmsd:.2f} A")

    lig_aligned = transform.apply(pred_lig)
    lig_aligned.res_name[:] = "LIG"
    tmp = out_sdf.replace('.sdf', '_tmp.pdb')
    f = pdb.PDBFile(); f.set_structure(lig_aligned); f.write(tmp)
    m = Chem.MolFromPDBFile(tmp, sanitize=False, removeHs=False)
    if m is None:
        raise ValueError("RDKit could not read aligned ligand")

    tmpl = Chem.MolFromMolFile(template_sdf)
    if tmpl is None:
        raise ValueError("could not load template")
    m = AllChem.AssignBondOrdersFromTemplate(tmpl, m)
    Chem.SanitizeMol(m)

    w = Chem.SDWriter(out_sdf); w.write(m); w.close()
    os.remove(tmp)
    c = m.GetConformer().GetPositions()
    print(f"  wrote {out_sdf}  centroid={np.round(c.mean(axis=0), 2)}")

if __name__ == "__main__":
    run(*sys.argv[1:])
