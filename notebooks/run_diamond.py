"""Optional train-only DIAMOND top-hit transfer; invoke only from notebook 02."""
import csv,json,subprocess,sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from Bio import SeqIO
from protfunc.data import parse_terms,parse_obo,propagate_labels,exclude_root_terms,exclude_root_labels
from protfunc.eval import aspect_metrics, METRIC_COLUMNS

def main():
    data=Path(sys.argv[1]); prep=Path(sys.argv[2]); out=Path(sys.argv[3]);
    required=[data/"Train/train_sequences.fasta",data/"Train/train_terms.tsv",data/"Train/go-basic.obo",data/"Train/train_taxonomy.tsv",prep/"splits.json"]
    missing=[str(p) for p in required if not p.is_file()]
    if missing: raise FileNotFoundError("Missing configured file(s): "+", ".join(missing))
    if subprocess.run(["diamond","version"],capture_output=True).returncode: raise RuntimeError("DIAMOND is not installed; run the optional install cell first")
    splits=json.loads((prep/"splits.json").read_text())["splits"]
    terms=exclude_root_terms(json.loads((prep/"term_lists.json").read_text()))
    seqs={r.id:str(r.seq).upper() for r in SeqIO.parse(required[0],"fasta") if len(r.seq)<=512}
    train=set(splits["train"]); test=list(splits["test"]); test_set=set(test); annotations=parse_terms(required[1]); parents,aspects=parse_obo(required[2]); annotations,parents=exclude_root_labels(annotations,parents); labels=propagate_labels(annotations,parents)
    flat=sum((terms[a] for a in ("MF","BP","CC")),[]); ti={t:i for i,t in enumerate(flat)}; offsets={}; n=0
    for a in ("MF","BP","CC"): offsets[a]=(n,n+len(terms[a])); n+=len(terms[a])
    wd=out/"diamond"; wd.mkdir(parents=True,exist_ok=True); train_fa=wd/"train.fasta"; test_fa=wd/"test.fasta"
    SeqIO.write([r for r in SeqIO.parse(required[0],"fasta") if r.id in train and r.id in seqs],train_fa,"fasta")
    SeqIO.write([r for r in SeqIO.parse(required[0],"fasta") if r.id in test_set],test_fa,"fasta")
    db=wd/"train_db"; subprocess.run(["diamond","makedb","--in",str(train_fa),"--db",str(db)],check=True)
    tab=wd/"hits.tsv"; subprocess.run(["diamond","blastp","--db",str(db),"--query",str(test_fa),"--out",str(tab),"--max-target-seqs","1","--outfmt","6","qseqid","sseqid","bitscore"],check=True)
    score=np.zeros((len(test),len(flat)),np.float32); test_i={p:i for i,p in enumerate(test)}
    for row in open(tab):
        q,s,bits=row.rstrip().split("\t"); weight=float(bits)/(float(bits)+100.0)
        for term in labels.get(s,()):
            if term in ti: score[test_i[q],ti[term]]=weight
    truth=np.zeros_like(score)
    for i,p in enumerate(test):
        for t in labels.get(p,()):
            if t in ti: truth[i,ti[t]]=1
    freq={t:sum(t in labels.get(pid,()) for pid in train) for t in flat}; rows=[]
    for a,(lo,hi) in offsets.items(): rows.append({"model":"diamond_top_hit","aspect":a,**aspect_metrics(truth[:,lo:hi],score[:,lo:hi],[freq[t] for t in terms[a]]),"split_method":json.loads((prep/"splits.json").read_text())["metadata"]["method"]})
    metrics=out/"results/metrics.csv"
    existing=list(csv.DictReader(open(metrics,newline="",encoding="utf-8"))) if metrics.exists() else []
    with open(metrics,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=METRIC_COLUMNS); w.writeheader(); w.writerows(existing+rows)
    all_rows=existing+rows
    plt.figure(figsize=(10,5)); plt.bar([r["model"]+"-"+r["aspect"] for r in all_rows],[float(r["fmax"]) for r in all_rows]); plt.xticks(rotation=45,ha="right"); plt.ylabel("F-max"); plt.tight_layout(); plt.savefig(out/"results/fmax_by_model_aspect.png",dpi=160)
    print("Appended measured DIAMOND metrics to",metrics)

if __name__=="__main__": main()
