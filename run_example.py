# -*- coding: utf-8 -*-
"""Representative single-replication figures for Section 4.1.3."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from simulation_core import (
    simulate_estimation_dgp, run_skater_path, partition_to_labels,
    recover_beta_curve, match_true_to_estimated,
    FVE_DEFAULT, K_NEIGHBORS_DEFAULT, MIN_SIZE_DEFAULT,
)


def _draw_graph(ax, coords, graph, labels=None):
    for u,v in graph.edges():
        ax.plot([coords[u,0],coords[v,0]],[coords[u,1],coords[v,1]],linewidth=0.45,alpha=0.45)
    if labels is None:
        ax.scatter(coords[:,0],coords[:,1],s=10)
    else:
        ax.scatter(coords[:,0],coords[:,1],c=labels,s=12)
    ax.set_xlabel("u"); ax.set_ylabel("v")


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output-dir",type=Path,default=Path("results/example")); args=ap.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    data=[]; res=[]
    for design,seed in [("uniform",2026),("non-uniform",2027)]:
        d=simulate_estimation_dgp(spatial_design=design,error_distribution="normal_N025",seed=seed)
        r=run_skater_path(d,fve_threshold=FVE_DEFAULT,k_neighbors=K_NEIGHBORS_DEFAULT,min_size=MIN_SIZE_DEFAULT,K_max=None)
        data.append(d); res.append(r)

    # Figure 1
    fig,axes=plt.subplots(1,2,figsize=(8,4))
    for ax,d,title in zip(axes,data,["(a) Uniform design","(b) Non-uniform design"]):
        ax.scatter(d["coords"][:,0],d["coords"][:,1],c=d["true_group"],s=12); ax.set_title(title); ax.set_xlabel("u"); ax.set_ylabel("v")
    fig.tight_layout(); fig.savefig(args.output_dir/"figure1_true_groups.png",dpi=300,bbox_inches="tight"); plt.close(fig)

    # Figure 4
    fig,axes=plt.subplots(2,2,figsize=(8,8))
    for r_idx,(d,r,name) in enumerate(zip(data,res,["Uniform","Non-uniform"])):
        _draw_graph(axes[r_idx,0],d["coords"],r["adjacency"]); axes[r_idx,0].set_title(f"{name}: adjacency graph")
        _draw_graph(axes[r_idx,1],d["coords"],r["mst"]); axes[r_idx,1].set_title(f"{name}: MST")
    fig.tight_layout(); fig.savefig(args.output_dir/"figure4_adjacency_mst.png",dpi=300,bbox_inches="tight"); plt.close(fig)

    # Figure 5
    fig,axes=plt.subplots(1,2,figsize=(8,4))
    for ax,d,r,title in zip(axes,data,res,["(a) Uniform design","(b) Non-uniform design"]):
        labels=partition_to_labels(r["best_record"]["partition"],len(d["true_group"]))
        _draw_graph(ax,d["coords"],r["mst"],labels); ax.set_title(title+f"; K={r['best_K']}")
    fig.tight_layout(); fig.savefig(args.output_dir/"figure5_selected_partitions.png",dpi=300,bbox_inches="tight"); plt.close(fig)

    # Figure 6 beta
    fig,axes=plt.subplots(2,1,figsize=(8,6),sharex=True)
    for ax,d,r,title in zip(axes,data,res,["(a) Uniform design","(b) Non-uniform design"]):
        pred=partition_to_labels(r["best_record"]["partition"],len(d["true_group"]))
        mapping=match_true_to_estimated(d["true_group"],pred)
        basis=r["prepared_data"]["fpca_basis"]
        for g in range(4):
            ax.plot(d["s_grid"],d["beta_curves"][g],linewidth=1.2,label=f"true beta g{g+1}")
            if g in mapping:
                th=r["best_record"]["group_results"][mapping[g]]["theta"]
                ax.plot(d["s_grid"],recover_beta_curve(th,basis),linestyle="--",linewidth=1.2,label=f"estimated beta g{g+1}")
        ax.set_ylabel(r"$\beta_g(s)$"); ax.set_title(title)
    axes[-1].set_xlabel("s"); axes[-1].legend(ncol=4,fontsize=7,frameon=False)
    fig.tight_layout(); fig.savefig(args.output_dir/"figure6_beta_curves.png",dpi=300,bbox_inches="tight"); plt.close(fig)

    # Figure 7 gamma
    fig,axes=plt.subplots(2,1,figsize=(9,6))
    for ax,d,r,title in zip(axes,data,res,["(a) Uniform design","(b) Non-uniform design"]):
        pred=partition_to_labels(r["best_record"]["partition"],len(d["true_group"]))
        mapping=match_true_to_estimated(d["true_group"],pred); M=r["best_record"]["M_fpca"]
        true=[]; est=[]; labels=[]
        for g in range(4):
            for j in range(3):
                true.append(d["gamma_true"][g,j]); labels.append(f"g{g+1},z{j+1}")
                est.append(r["best_record"]["group_results"][mapping[g]]["theta"][M+j] if g in mapping else np.nan)
        x=np.arange(len(true)); width=.38
        ax.bar(x-width/2,true,width,label="True gamma"); ax.bar(x+width/2,est,width,label="Estimated gamma")
        ax.set_xticks(x); ax.set_xticklabels(labels,rotation=45,ha="right",fontsize=7); ax.set_title(title); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(args.output_dir/"figure7_gamma.png",dpi=300,bbox_inches="tight"); plt.close(fig)
    print("Saved representative figures to",args.output_dir)

if __name__=="__main__": main()