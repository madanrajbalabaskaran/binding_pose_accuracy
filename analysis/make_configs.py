import os
import sys
import pandas as pd

COND = sys.argv[1]
W = "/nobackup/proj/comet_tsapbip/c5052585"
SYS = W + "/scale/systems"
MSA = W + "/scale/msa"
OUT = W + "/scale/configs/" + COND
os.makedirs(OUT, exist_ok=True)

def a3m_query(p):
    with open(p) as f:
        f.readline()
        return f.readline().strip()

d = pd.read_csv(W + "/ishitani/candidates.csv")
made = 0
skip = []

for _, r in d.iterrows():
    sid = r["sys_id"]
    a3m = "%s/%s.a3m" % (MSA, sid)
    if not os.path.isfile(a3m):
        skip.append(sid)
        continue
    seq = a3m_query(a3m)
    if not seq:
        skip.append(sid)
        continue

    L = []
    L.append("# %s %s" % (sid, r["ccd_id"]))
    L.append("sequences:")
    L.append("  - protein:")
    L.append("      id: A")
    L.append('      sequence: "%s"' % seq)
    L.append("      msa: %s" % a3m)
    L.append("  - ligand:")
    L.append("      id: B")
    L.append("      smiles: '%s'" % r["smiles"])

    if COND in ("ext", "fixed"):
        L.append("      chiral_restraints: true")
    if COND == "fixed":
        L.append("      pose_restraints: true")
        L.append("      pose_w_torsion: 1.0")
    if COND in ("ext", "fixed"):
        L.append("restraints_config:")
        L.append("  angle:")
        L.append("    weight: 1")
        L.append("  bond:")
        L.append("    weight: 1")
        L.append("  chiral:")
        L.append("    weight: 1")
        L.append("  start_sigma: 1.0")
        L.append("  gpu: false")
        L.append("  verbose: true")

    f = "%s/%s.yaml" % (OUT, sid)
    open(f, "w").write("\n".join(L) + "\n")
    made += 1

print(COND, "written:", made)
print("skipped:", skip)
