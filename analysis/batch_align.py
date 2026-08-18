import glob
import os
import re
import subprocess
import pandas as pd

W = "/nobackup/proj/comet_tsapbip/c5052585"
S = W + "/scale"
OUT = S + "/aligned"
d = pd.read_csv(W + "/ishitani/candidates.csv")

rows = []
for cond in ("control", "ext", "fixed", "fixed_s138", "fixed_s408"):
    os.makedirs("%s/%s" % (OUT, cond), exist_ok=True)
    for sid in d["sys_id"]:
        lig = sid.split("__")[3]
        tmpl = "%s/systems/%s/ligand_files/%s.sdf" % (S, sid, lig)
        rec = "%s/systems/%s/receptor.pdb" % (S, sid)
        pat = "%s/out/%s/%s/**/*_model_*.cif" % (S, cond, sid)
        cifs = sorted(glob.glob(pat, recursive=True))
        if not cifs or not os.path.isfile(tmpl):
            rows.append({"cond": cond, "sys_id": sid, "sample": -1,
                         "sdf": "", "prot_rmsd": None, "status": "NO INPUT"})
            continue
        for cif in cifs:
            i = int(re.search(r"_model_(\d+)\.cif$", cif).group(1))
            out = "%s/%s/%s__%s__m%d.sdf" % (OUT, cond, cond, sid, i)
            r = subprocess.run(
                ["python", os.path.join(os.path.dirname(os.path.abspath(__file__)), "align_and_fix_ligand.py"),
                 cif, "B", "A", rec, tmpl, out],
                capture_output=True, text=True)
            m = re.search(r"protein RMSD = ([\d.]+)", r.stdout)
            rows.append({
                "cond": cond, "sys_id": sid, "sample": i, "sdf": out,
                "prot_rmsd": float(m.group(1)) if m else None,
                "status": "OK" if os.path.isfile(out) else "FAIL",
            })

t = pd.DataFrame(rows)
t.to_csv(S + "/align_manifest.csv", index=False)
print(t["status"].value_counts().to_string())
print()
print("protein RMSD > 1.0 A:", int((t["prot_rmsd"] > 1.0).sum()))
bad = t[(t["status"] != "OK") | (t["prot_rmsd"] > 1.0)]
cols = ["cond", "sys_id", "sample", "prot_rmsd", "status"]
print(bad[cols].head(30).to_string(index=False))
