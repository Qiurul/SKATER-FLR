# -*- coding: utf-8 -*-
"""Paper-aligned Monte Carlo estimation experiment."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from .simulation_core import (
    simulate_estimation_dgp, run_skater_path, estimation_metrics,
    FVE_DEFAULT, K_NEIGHBORS_DEFAULT, MIN_SIZE_DEFAULT,
)

DESIGNS=("uniform","non-uniform")
ERRORS=("normal_N025","t8_scaled","uniform_var025")

def one_rep(design,error,rep):
    seed=20260000+1000*DESIGNS.index(design)+100*ERRORS.index(error)+rep
    d=simulate_estimation_dgp(spatial_design=design,error_distribution=error,seed=seed)
    r=run_skater_path(d,fve_threshold=FVE_DEFAULT,k_neighbors=K_NEIGHBORS_DEFAULT,min_size=MIN_SIZE_DEFAULT,K_max=None)
    m=estimation_metrics(d,r)
    return {"spatial_design":design,"error_distribution":error,"rep":rep,**m}

def summarize(df):
    metrics=[c for c in df.columns if c not in {"spatial_design","error_distribution","rep"}]
    rows=[]
    for (d,e),g in df.groupby(["spatial_design","error_distribution"],sort=False):
        row={"spatial_design":d,"error_distribution":e,"reps":len(g)}
        for c in metrics:
            if pd.api.types.is_numeric_dtype(g[c]):
                row[c+"_mean"]=g[c].mean()
                row[c+"_sd"]=g[c].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--reps",type=int,default=100)
    ap.add_argument("--jobs",type=int,default=1)
    ap.add_argument("--output-dir",type=Path,default=Path("outputs/estimation"))
    args=ap.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    tasks=[(d,e,r) for d in DESIGNS for e in ERRORS for r in range(args.reps)]
    rows=Parallel(n_jobs=args.jobs,verbose=5)(delayed(one_rep)(*t) for t in tasks)
    raw=pd.DataFrame(rows); raw.to_csv(args.output_dir/"estimation_replications.csv",index=False)
    summary=summarize(raw); summary.to_csv(args.output_dir/"estimation_summary.csv",index=False)
    print(summary.to_string(index=False))

if __name__=="__main__": main()
