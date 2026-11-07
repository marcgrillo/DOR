import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, so `import bdor` works without installing

import json, glob, os, re, itertools
import numpy as np

from bdor.paths import DATASETS_DIR

pat = re.compile(r"grid_F1=(\d+)_F2=(\d+)_S=(\d+)_AR=(\d+)_K=(\d+)_Inc=(\d+)$")
folders = [d for d in glob.glob(os.path.join(DATASETS_DIR, "grid_*")) if os.path.isdir(d)]

# accumulate flips per Inc level
from collections import defaultdict
allpairs = defaultdict(lambda: [0,0])   # inc -> [flipped, total]
adjpairs = defaultdict(lambda: [0,0])
percards = defaultdict(lambda: [0,0])   # card-count mismatch vs true card gap
bylevel_byk = defaultdict(lambda: [0,0])

for d in folders:
    m = pat.search(os.path.basename(d))
    if not m: continue
    f1,f2,s,ar,k,inc = map(int, m.groups())
    if ar != k:      # AR==K filter
        continue
    js = glob.glob(os.path.join(d,"*.json"))
    if not js: continue
    data = json.load(open(js[0]))
    nmax = data['metadata']['n_max']
    for run in data['runs']:
        tr = run['true_ranks']          # tr[a] = rank (0 best)
        tu = run['true_utilities']
        for sr in run['subranks']:
            items = sr['items']
            kk = len(items)
            # all declared pairwise comparisons (position p<q => elicited prefers items[p])
            for p in range(kk):
                for q in range(p+1,kk):
                    a,b = items[p], items[q]
                    flip = 1 if tr[a] > tr[b] else 0
                    allpairs[inc][0]+=flip; allpairs[inc][1]+=1
                    bylevel_byk[(inc,k)][0]+=flip; bylevel_byk[(inc,k)][1]+=1
                    if q==p+1:
                        adjpairs[inc][0]+=flip; adjpairs[inc][1]+=1

print("=== Realized FLIPPED pairwise comparisons (AR==K datasets) ===")
print(f"{'TargetInc':>9} | {'all-pairs flip%':>15} | {'adjacent flip%':>15}")
for inc in sorted(allpairs):
    ap = 100*allpairs[inc][0]/allpairs[inc][1]
    aj = 100*adjpairs[inc][0]/adjpairs[inc][1]
    print(f"{inc:>9} | {ap:>15.2f} | {aj:>15.2f}   (n_pairs={allpairs[inc][1]})")

print("\n=== all-pairs flip% by level and subset size k ===")
for inc in sorted(set(i for i,_ in bylevel_byk)):
    row=[]
    for k in [3,4,5]:
        f,t=bylevel_byk[(inc,k)]
        row.append(f"k={k}:{100*f/t:.2f}%")
    print(f"Inc={inc}: "+"  ".join(row))
