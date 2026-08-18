import os

import numpy as np

import pandas as pd



S = "/nobackup/proj/comet_tsapbip/c5052585/scale"

d = pd.read_csv(S + "/pb_scale.csv")



def parse(p):

    b = os.path.basename(str(p)).replace(".sdf", "")

    q = b.split("__")

    return q[0], "__".join(q[1:5]), int(q[5][1:])



d[["cond", "sys_id", "sample"]] = pd.DataFrame(

    [parse(f) for f in d["file"]], index=d.index)



m = pd.read_csv(S + "/align_manifest.csv")

m = m[["cond", "sys_id", "sample", "prot_rmsd"]]

d = d.merge(m, on=["cond", "sys_id", "sample"], how="left")



ORDER = ["control", "ext", "fixed", "fixed_s138", "fixed_s408"]

d["cond"] = pd.Categorical(d["cond"], categories=ORDER, ordered=True)



print("poses scored:", len(d))

print(d.groupby("cond", observed=True).size().to_string(), "\n")



# exclusions

pr = d.groupby("sys_id")["prot_rmsd"].mean()

drop = sorted(pr[pr > 1.0].index)

print("EXCLUDED (mean protein RMSD > 1.0 A):", len(drop))

print(pr[pr > 1.0].round(2).to_string(), "\n")

k = d[~d["sys_id"].isin(drop)].copy()

print("systems retained:", k["sys_id"].nunique(), "\n")



NUM = ["rmsd", "kabsch_rmsd", "mol_pred_energy", "energy_ratio"]

for c in NUM:

    if c in k.columns:

        print("---", c)

        print(k.groupby("cond", observed=True)[c]

               .agg(["mean", "std", "count"]).round(3).to_string(), "\n")

CHECKS = ["tetrahedral_chirality", "stereochemistry_preserved",
          "bond_lengths", "bond_angles", "internal_steric_clash"]
for c in k.columns:
    if "rmsd" in str(c) and "2" in str(c):
        CHECKS.append(c)

print("--- pass rates")
rows = {}
for chk in CHECKS:
    if chk not in k.columns:
        continue
    try:
        rows[chk] = {c: "%d/%d" % (int(v.sum()), len(v))
                     for c, v in k.groupby("cond", observed=True)[chk]}
    except Exception:
        pass
print(pd.DataFrame(rows).T.to_string(), "\n")



# system-level paired comparison

piv = k.groupby(["sys_id", "cond"], observed=True)[NUM].mean().reset_index()

print("--- paired system-level contrasts")

for c in NUM:

    w = piv.pivot(index="sys_id", columns="cond", values=c).dropna()

    print("==", c, " n =", len(w))

    for a, b in [("control", "ext"), ("ext", "fixed"), ("control", "fixed")]:

        dl = w[b] - w[a]

        print("   %-8s -> %-8s  mean %+7.3f   %s better %d/%d" %

              (a, b, dl.mean(), b, int((dl < 0).sum()), len(dl)))

    print()



w = piv.pivot(index="sys_id", columns="cond", values="rmsd").dropna()

hard = w[w["control"] > 2.0].index

print("--- stratified by control difficulty")

print("control fails (>2 A):", len(hard), " succeeds:", len(w) - len(hard))

for name, idx in [("HARD", hard), ("EASY", w.index.difference(hard))]:

    if len(idx) == 0:

        continue

    sub = w.loc[idx]

    print("%s n=%d  control %.3f  ext %.3f  fixed %.3f" %

          (name, len(sub), sub["control"].mean(),

           sub["ext"].mean(), sub["fixed"].mean()))
