import os
import subprocess
import pandas as pd

W = "/nobackup/proj/comet_tsapbip/c5052585"
ZIPS = W + "/plinder/2024-06/v2/systems"
DEST = W + "/scale/systems"
os.makedirs(DEST, exist_ok=True)

d = pd.read_csv(W + "/ishitani/candidates.csv")
ok, bad = 0, []
for _, r in d.iterrows():
    sid, pdb = r["sys_id"], r["pdb_id"]
    shard = pdb[1:3]
    if os.path.isdir(DEST + "/" + sid):
        ok += 1
        continue
    z = "%s/%s.zip" % (ZIPS, shard)
    subprocess.run(["unzip", "-o", "-q", z, sid + "/*", "-d", DEST],
                   capture_output=True, text=True)
    if os.path.isdir(DEST + "/" + sid):
        ok += 1
    else:
        bad.append((sid, shard))

print("extracted:", ok, "/", len(d))
print("failed:", bad)
