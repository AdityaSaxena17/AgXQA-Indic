"""Plot only saved measurements; explicitly mark unrecorded validation loss."""
import argparse
import csv
import json
from pathlib import Path
import os

from _common import PHASE
os.environ.setdefault("MPLCONFIGDIR", str(PHASE / ".cache/matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PHASE / ".cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir",type=Path,default=PHASE/"results")
    args=parser.parse_args()
    root=args.results_dir
    output=root/"plots"
    output.mkdir(parents=True,exist_ok=True)
    history=list(csv.DictReader((root/"training_logs.csv").open()))
    epochs=[int(r["epoch"]) for r in history]
    plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False,"figure.dpi":150})
    fig,ax=plt.subplots(figsize=(8,4.5))
    ax.plot(epochs,[float(r["training_loss"]) for r in history],color="#276f9b",marker="o",linewidth=2)
    ax.set(xlabel="Epoch",ylabel="Mean training cross-entropy loss",title="AgRoBERTa reimplementation: measured training loss",xticks=[1,5,10,15,20])
    ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(output/"training_loss.png"); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5))
    if all(r["validation_loss"] for r in history):
        ax.plot(epochs,[float(r["validation_loss"]) for r in history],marker="o")
        ax.set(xlabel="Epoch",ylabel="Validation cross-entropy loss",title="Measured validation loss")
    else:
        ax.axis("off")
        ax.text(.5,.72,"Validation loss was not recorded",ha="center",va="center",fontsize=18,weight="bold",transform=ax.transAxes)
        ax.text(.5,.43,"The original run saved training loss and validation EM/F1.\nNo validation-loss values or curve are available.\nThis image is a data-availability notice, not a measurement.",ha="center",va="center",fontsize=12,linespacing=1.6,transform=ax.transAxes)
    fig.tight_layout(); fig.savefig(output/"validation_loss.png"); plt.close(fig)
    summary=json.loads((root/"baseline_results.json").read_text())
    series=[summary["published_reference"],summary["before_finetuning"],summary["agroberta_reimplementation"]]
    x=np.arange(3); width=.35
    fig,ax=plt.subplots(figsize=(9,5))
    for offset,key,label,color in ((-width/2,"exact_match","EM","#276f9b"),(width/2,"f1","Token F1","#bf6b35")):
        bars=ax.bar(x+offset,[r[key] for r in series],width,label=label,color=color)
        bars[0].set_hatch("//"); bars[0].set_alpha(.65)
        ax.bar_label(bars,fmt="%.2f",padding=3,fontsize=10)
    ax.set(xticks=x,xticklabels=["Paper\n(published reference)","Agricultural backbone + SQuAD\n(measured before fine-tuning)","AgRoBERTa reimplementation\n(measured, 20 epochs)"],ylabel="Score (%)",ylim=(0,100),title="English AgXQA test: SQuAD-style EM and F1")
    ax.legend(loc="upper left");ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True)
    fig.text(.5,.02,"330 test questions. XLM-R, RoBERTa Base, BERT and DistilBERT were not run.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.05,1,1));fig.savefig(output/"baseline_comparison.png");plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5))
    ax.plot(epochs,[float(r["validation_f1"]) for r in history],label="Validation F1",marker="o",color="#bf6b35")
    ax.plot(epochs,[float(r["validation_exact_match"]) for r in history],label="Validation EM",marker="s",color="#276f9b")
    ax.set(xlabel="Epoch",ylabel="Score (%)",title="Validation metrics: all 353 questions",ylim=(40,85),xticks=[1,5,10,15,20])
    ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(output/"f1_across_epochs.png");plt.close(fig)
    stats=json.loads((root/"dataset_statistics.json").read_text())
    names=["train","validation","test"]
    counts=[stats["audit"]["splits"][k]["rows"] for k in names]
    fig,ax=plt.subplots(figsize=(7,4.5))
    bars=ax.bar(["Train","Validation","Test"],counts,color=["#276f9b","#bf6b35","#54856b"])
    ax.bar_label(bars,padding=4);ax.set(ylabel="QA pairs",title="Pinned published English AgXQA splits",ylim=(0,1700))
    fig.tight_layout();fig.savefig(output/"dataset_distribution.png");plt.close(fig)
    print(f"Saved measured plots and the explicit missing-loss notice in {output}")


if __name__ == "__main__":
    main()
