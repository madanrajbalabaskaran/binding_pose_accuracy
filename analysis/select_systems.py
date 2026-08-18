import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
RDLogger.DisableLog('rdApp.*')

BASE = "/nobackup/proj/comet_tsapbip/c5052585/plinder/2024-06/v2"
WORK = "/nobackup/proj/comet_tsapbip/c5052585/ishitani"

d = pd.read_csv(WORK + "/plinder_l70.csv")
d = d[d["afterP70L70"]].copy()
print("afterP70L70:", len(d))

rows = []
for _, r in d.iterrows():
    m = Chem.MolFromSmiles(r["smiles"])
    if m is None:
        continue
    cc = Chem.FindMolChiralCenters(m, useLegacyImplementation=False)
    rows.append({
        "sys_id": r["sys_id"],
        "pdb_id": r["pdb_id"],
        "ccd_id": r["ccd_id"],
        "smiles": r["smiles"],
        "heavy": m.GetNumHeavyAtoms(),
        "rot": rdMolDescriptors.CalcNumRotatableBonds(m),
        "chiral": len(cc),
    })

p = pd.DataFrame(rows)
print("parsed:", len(p))

cols = ["system_id", "entry_resolution",
        "system_proper_num_protein_chains",
        "system_proper_num_ligand_chains"]
ann = pd.read_parquet(BASE + "/index/annotation_table.parquet",
                      columns=cols)
ann = ann.drop_duplicates("system_id")
ann = ann.rename(columns={"system_id": "sys_id"})
p = p.merge(ann, on="sys_id", how="left")
print("with annotations:", int(p["entry_resolution"].notna().sum()))

sel = p[(p["heavy"] >= 15) & (p["heavy"] <= 45)
        & (p["rot"] >= 3)
        & (p["chiral"] >= 1)
        & (p["entry_resolution"] <= 2.5)
        & (p["system_proper_num_ligand_chains"] == 1)].copy()
sel = sel.sort_values("entry_resolution")

print()
print("SELECTED:", len(sel))
show = ["sys_id", "ccd_id", "heavy", "rot", "chiral",
        "entry_resolution", "system_proper_num_protein_chains"]
print(sel[show].head(60).to_string(index=False))
sel.to_csv(WORK + "/candidates.csv", index=False)
print()
print("wrote", WORK + "/candidates.csv")
