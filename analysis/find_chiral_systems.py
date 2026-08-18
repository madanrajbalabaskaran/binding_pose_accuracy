import pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

base = '/nobackup/proj/comet_tsapbip/c5052585/plinder/2024-06/v2'
ann   = pd.read_parquet(f'{base}/index/annotation_table.parquet')
split = pd.read_parquet(f'{base}/splits/split.parquet')

test = set(split[split['split'] == 'test']['system_id'])

df = ann[ann['system_id'].isin(test)
       & (ann['system_proper_num_ligand_chains'] == 1)
       & (ann['system_proper_num_protein_chains'] == 1)
       & (~ann['ligand_is_covalent'])
       & (~ann['ligand_is_ion'])
       & (~ann['ligand_is_fragment'])
       & (~ann['ligand_is_artifact'])
       & (ann['ligand_num_heavy_atoms'].between(15, 35))
       & (ann['ligand_num_rot_bonds'] >= 3)
       & (ann['entry_resolution'] < 2.0)
       & (pd.to_datetime(ann['entry_release_date']) > '2021-09-30')
       ].drop_duplicates('system_id').sort_values('entry_resolution')

print(f"{len(df)} systems pass structural filters; checking chirality...\n")

hits = []
for _, r in df.iterrows():
    m = Chem.MolFromSmiles(r['ligand_rdkit_canonical_smiles'])
    if m is None:
        continue
    centres = Chem.FindMolChiralCenters(
        m, includeUnassigned=True, useLegacyImplementation=False)
    if len(centres) > 0:
        hits.append(dict(
            system_id = r['system_id'],
            ccd       = r['ligand_ccd_code'],
            chiral    = len(centres),
            rot       = int(r['ligand_num_rot_bonds']),
            heavy     = int(r['ligand_num_heavy_atoms']),
            res       = round(float(r['entry_resolution']), 2),
            released  = str(r['entry_release_date'])[:10],
            prot_len  = r['system_protein_chains_length'],
            smiles    = r['ligand_rdkit_canonical_smiles'],
        ))
    if len(hits) >= 15:
        break

out = pd.DataFrame(hits)
pd.set_option('display.width', 220)
pd.set_option('display.max_colwidth', 45)
print(out.to_string(index=False))
