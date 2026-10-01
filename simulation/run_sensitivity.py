# -*- coding: utf-8 -*-
"""One-factor-at-a-time sensitivity analysis for FVE, kNN, and minimum group size."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from joblib import Parallel, delayed
from .simulation_core import (
    simulate_estimation_dgp, run_skater_path, estimation_metrics,
    FVE_DEFAULT, K_NEIGHBORS_DEFAULT, MIN_SIZE_DEFAULT,
)

GRIDS={"FVE":[.80,.85,.90,.95],"kNN":[4,5,6,7,8],"min_size":[5,10,15,20,25]}

def one_rep(factor,value,rep,design,error):
    d=simulate_estimation_dgp(spatial_design=design,error_distribution=error,seed=20261000+rep)
    kw=dict(fve_threshold=FVE_DEFAULT,k_neighbors=K_NEIGHBORS_DEFAULT,min_size=MIN_SIZE_DEFAULT,K_max=None)
    if factor=="FVE": kw["fve_threshold"]=float(value)
    elif factor=="kNN": kw["k_neighbors"]=int(value)
    else: kw["min_size"]=int(value)
    r=run_skater_path(d,**kw)
    return {"factor":factor,"value":value,"rep":rep,"spatial_design":design,"error_distribution":error,**estimation_metrics(d,r)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--reps",type=int,default=100); ap.add_argument("--jobs",type=int,default=1)
    ap.add_argument("--design",choices=["uniform","non-uniform"],default="uniform")
    ap.add_argument("--error",choices=["normal_N025","t8_scaled","uniform_var025"],default="normal_N025")
    ap.add_argument("--output-dir",type=Path,default=Path("outputs/sensitivity"))
    args=ap.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    tasks=[(f,v,r,args.design,args.error) for f,vals in GRIDS.items() for v in vals for r in range(args.reps)]
    rows=Parallel(n_jobs=args.jobs,verbose=5)(delayed(one_rep)(*t) for t in tasks)
    raw=pd.DataFrame(rows); raw.to_csv(args.output_dir/"sensitivity_replications.csv",index=False)
    num=[c for c in raw.columns if c not in {"factor","value","rep","spatial_design","error_distribution"}]
    summary=raw.groupby(["factor","value"],as_index=False)[num].mean(numeric_only=True)
    summary.to_csv(args.output_dir/"sensitivity_summary.csv",index=False); print(summary.to_string(index=False))

if __name__=="__main__": main()
