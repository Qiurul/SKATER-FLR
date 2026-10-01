# -*- coding: utf-8 -*-
"""Reproduce Section 4.2 empirical size and power of the joint beta/gamma test."""
from __future__ import annotations
import argparse
from pathlib import Path
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from .simulation_core import SPATIAL_DESIGNS, ERROR_DISTRIBUTIONS, TEST_C_VALUES, FVE_DEFAULT, K_NEIGHBORS_DEFAULT, MIN_SIZE_DEFAULT, simulate_test_dgp, joint_homogeneity_test

def parse_c_values(text): return tuple(float(x.strip()) for x in text.split(",") if x.strip())

def one_rep(c,c_idx,design,d_idx,error,e_idx,rep,B,start_seed):
    seed=int(start_seed+10_000_000*c_idx+100_000*d_idx+10_000*e_idx+rep); bseed=seed+50_000_000
    try:
        data=simulate_test_dgp(c=c,spatial_design=design,error_distribution=error,seed=seed)
        test=joint_homogeneity_test(data,B=B,random_state=bseed,fve_threshold=FVE_DEFAULT,k_neighbors=K_NEIGHBORS_DEFAULT,min_size=MIN_SIZE_DEFAULT,K_max=None)
        return {"c":c,"spatial_design":design,"error_distribution":error,"rep":rep+1,"data_seed":seed,"bootstrap_seed":bseed,"T_obs":test["T_obs"],"K_alt":test["K_alt"],"B_valid":test["B_valid"],"B_failed":test["B_failed"],"p_value":test["p_value"],"status":"ok"}
    except Exception as exc:
        return {"c":c,"spatial_design":design,"error_distribution":error,"rep":rep+1,"data_seed":seed,"bootstrap_seed":bseed,"p_value":np.nan,"status":f"{type(exc).__name__}: {exc}"}

def chunk_complete(path,n):
    if not path.exists(): return False
    try: df=pd.read_csv(path)
    except Exception: return False
    return len(df)==n and df.get("status",pd.Series(dtype=str)).astype(str).eq("ok").all() and df["p_value"].notna().all()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--c-values",default=",".join(str(x) for x in TEST_C_VALUES)); ap.add_argument("--reps",type=int,default=200); ap.add_argument("--B",type=int,default=1000); ap.add_argument("--jobs",type=int,default=1); ap.add_argument("--batch-size",type=int,default=50); ap.add_argument("--start-seed",type=int,default=7000); ap.add_argument("--output-dir",type=Path,default=Path("outputs/joint_test"))
    args=ap.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True); chunks=args.output_dir/"chunks"; chunks.mkdir(exist_ok=True); c_values=parse_c_values(args.c_values)
    all_paths=[]
    for c_idx,c in enumerate(c_values):
      for d_idx,design in enumerate(SPATIAL_DESIGNS):
       for e_idx,error in enumerate(ERROR_DISTRIBUTIONS):
        for start in range(0,args.reps,args.batch_size):
         end=min(start+args.batch_size,args.reps); n=end-start; name=f"c_{c:.2f}_{design}_{error}_rep_{start+1:04d}_{end:04d}.csv".replace(".","p",1); path=chunks/name; all_paths.append(path)
         if chunk_complete(path,n): print("[SKIP]",path.name); continue
         tasks=list(range(start,end)); print(f"[RUN] c={c} design={design} error={error} reps={start+1}-{end} B={args.B}"); tic=time.time()
         if args.jobs==1: rows=[one_rep(c,c_idx,design,d_idx,error,e_idx,r,args.B,args.start_seed) for r in tasks]
         else:
          with parallel_config(backend="loky",inner_max_num_threads=1): rows=Parallel(n_jobs=min(args.jobs,n),batch_size=1)(delayed(one_rep)(c,c_idx,design,d_idx,error,e_idx,r,args.B,args.start_seed) for r in tasks)
         pd.DataFrame(rows).to_csv(path,index=False); print(f"[SAVE] {path.name} ({(time.time()-tic)/60:.1f} min)")
    raw=pd.concat([pd.read_csv(p) for p in all_paths],ignore_index=True).drop_duplicates(["c","spatial_design","error_distribution","rep"],keep="last"); raw.to_csv(args.output_dir/"joint_test_raw.csv",index=False)
    valid=raw[(raw["status"]=="ok")&raw["p_value"].notna()].copy()
    size=valid[np.isclose(valid["c"],1.0)].groupby(["spatial_design","error_distribution"],as_index=False).agg(alpha_001=("p_value",lambda x:float(np.mean(x<.01))),alpha_005=("p_value",lambda x:float(np.mean(x<.05))),alpha_010=("p_value",lambda x:float(np.mean(x<.10))),valid_reps=("p_value","size")); size.to_csv(args.output_dir/"table3_size.csv",index=False)
    power=(valid.groupby(["spatial_design","error_distribution","c"],as_index=False).agg(rejection_rate=("p_value",lambda x:float(np.mean(x<.05))),valid_reps=("p_value","size"))); power.to_csv(args.output_dir/"table4_power_long.csv",index=False)
    wide=power.pivot(index=["spatial_design","error_distribution"],columns="c",values="rejection_rate").reset_index(); wide.to_csv(args.output_dir/"table4_power.csv",index=False)
    print("\nTable 3 (size):\n",size.to_string(index=False)); print("\nTable 4 (power):\n",wide.to_string(index=False))
if __name__=="__main__": main()
