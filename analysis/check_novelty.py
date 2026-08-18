import sys
import pandas as pd

base = '/nobackup/proj/comet_tsapbip/c5052585/plinder/2024-06/v2'
ann   = pd.read_parquet(f'{base}/index/annotation_table.parquet')
split = pd.read_parquet(f'{base}/splits/split.parquet')

for sid in sys.argv[1:]:
    row = ann[ann['system_id'] == sid].iloc[0]
    print(f"\n########## {sid}  (ccd={row['ligand_ccd_code']}) ##########")
    for thresh in ['50', '70', '95', '100']:
        col = f'tanimoto_similarity_max__{thresh}__community'
        cid = row[col]
        members = ann[ann[col] == cid]['system_id'].unique()
        counts = split[split['system_id'].isin(members)]['split'].value_counts()
        n_train = int(counts.get('train', 0))
        verdict = "NOVEL (no train analog)" if n_train == 0 else f"{n_train} train analogs"
        print(f"  {thresh:>3}%  cluster={cid:<10} size={len(members):<5} -> {verdict}")
