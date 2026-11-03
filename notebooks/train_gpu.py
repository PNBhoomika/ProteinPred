"""GPU notebook entry point. Run only inside Kaggle/Colab, never on the laptop."""
import argparse, csv, json, random, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from Bio import SeqIO
from protfunc.data import parse_terms, parse_obo, propagate_labels, select_terms, exclude_root_terms, exclude_root_labels
from protfunc.models import MultiHeadMLP, export_model
from protfunc.eval import aspect_metrics, hierarchy_enforce, METRIC_COLUMNS

def main():
    p=argparse.ArgumentParser(); p.add_argument("--data-dir",default="/kaggle/input/competitions/cafa-5-protein-function-prediction"); p.add_argument("--prep-dir",default="/kaggle/working/protfunc"); p.add_argument("--output-dir",default="/kaggle/working"); p.add_argument("--epochs",type=int,default=20); args=p.parse_args()
    d=Path(args.data_dir); prep=Path(args.prep_dir); out=Path(args.output_dir); (out/"models").mkdir(parents=True,exist_ok=True); (out/"results").mkdir(parents=True,exist_ok=True)
    required=[d/"Train/train_sequences.fasta",d/"Train/train_terms.tsv",d/"Train/go-basic.obo",d/"Train/train_taxonomy.tsv",prep/"splits.json"]
    missing=[str(x) for x in required if not x.is_file()]
    if missing: raise FileNotFoundError("Missing required input(s): "+", ".join(missing))
    seed=42; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    packed=json.loads((prep/"splits.json").read_text()); splits=packed["splits"]
    term_path=prep/"term_lists.json"
    term_lists=exclude_root_terms(json.loads(term_path.read_text())) if term_path.exists() else None
    parents,aspect=parse_obo(d/"Train/go-basic.obo"); annotations=parse_terms(d/"Train/train_terms.tsv")
    annotations,parents=exclude_root_labels(annotations,parents)
    ids=[]; shards=[]
    for f in sorted((prep/"embeddings").glob("emb_*.npy")):
        ids.extend(json.loads((prep/"embeddings"/f.name.replace("emb_","ids_").replace(".npy",".json")).read_text())); shards.append(np.load(f).astype(np.float32))
    if not shards: raise FileNotFoundError("No embedding shards under "+str(prep/"embeddings"))
    sampled=set(ids); labels=propagate_labels({p:annotations.get(p,set()) for p in sampled},parents)
    if term_lists is None: term_lists=exclude_root_terms(select_terms(labels,aspect,top_k=200))
    terms=sum((term_lists[a] for a in ("MF","BP","CC")),[]); index={t:i for i,t in enumerate(terms)}
    X=np.concatenate(shards); lookup={p:i for i,p in enumerate(ids)}; Y=np.zeros((len(ids),len(terms)),np.float32)
    for i,pid in enumerate(ids):
        for term in labels.get(pid,()):
            if term in index: Y[i,index[term]]=1
    pairs=set()
    for child in terms:
        todo=list(parents.get(child,())); seen=set()
        while todo:
            parent=todo.pop()
            if parent in seen: continue
            seen.add(parent)
            if parent in index: pairs.add((index[parent],index[child]))
            todo.extend(parents.get(parent,()))
    pairs=np.asarray(sorted(pairs),dtype=np.int64).reshape(-1,2)
    def ix(split): return np.asarray([lookup[x] for x in splits[split] if x in lookup],dtype=int)
    train_i,val_i,test_i=(ix(k) for k in ("train","val","test"))
    if min(map(len,(train_i,val_i,test_i)))==0: raise ValueError("A split has no embeddings")
    device="cuda" if torch.cuda.is_available() else "cpu"
    if device!="cuda": raise RuntimeError("Enable GPU acceleration; training is notebook-only")
    def fit(hier):
        model=MultiHeadMLP(X.shape[1],term_lists,512,hier,pairs).to(device); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4); best=float("inf"); state=None
        for epoch in range(args.epochs):
            model.train(); order=np.random.permutation(train_i)
            for start in range(0,len(order),512):
                batch=order[start:start+512]; xb=torch.as_tensor(X[batch],device=device); yb=torch.as_tensor(Y[batch],device=device); opt.zero_grad(); loss=F.binary_cross_entropy_with_logits(model(xb),yb); loss.backward(); opt.step()
            model.eval()
            with torch.no_grad(): val=F.binary_cross_entropy_with_logits(model(torch.as_tensor(X[val_i],device=device)),torch.as_tensor(Y[val_i],device=device)).item()
            print(("hierarchy" if hier else "flat"),epoch+1,"val_bce",val)
            if val<best: best=val; state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        model.load_state_dict(state); return model
    flat=fit(False); hier=fit(True); rows=[]; name_map={}; current=None
    for line in open(d/"Train/go-basic.obo",encoding="utf-8"):
        line=line.strip()
        if line=="[Term]": current=None
        elif line.startswith("id: GO:"): current=line[4:]
        elif current and line.startswith("name: "): name_map[current]=line[6:]
    offsets={}; offset=0
    for a in ("MF","BP","CC"): offsets[a]=(offset,offset+len(term_lists[a])); offset+=len(term_lists[a])
    train_ids=set(splits["train"]); frequencies={t:sum(t in labels.get(pid,()) for pid in train_ids) for t in terms}
    for name,model in (("flat",flat),("flat_posthoc",flat),("hierarchy",hier)):
        model.eval()
        with torch.no_grad(): logits=model(torch.as_tensor(X[test_i],device=device)).cpu().numpy()
        score=1/(1+np.exp(-logits))
        if name=="flat_posthoc": score=hierarchy_enforce(score,pairs)
        for a,(lo,hi) in offsets.items():
            metric=aspect_metrics(Y[test_i,lo:hi],score[:,lo:hi],[frequencies[t] for t in term_lists[a]])
            rows.append({"model":name,"aspect":a,**metric,"split_method":packed["metadata"]["method"]})
    with open(out/"results/metrics.csv","w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=METRIC_COLUMNS); w.writeheader(); w.writerows(rows)
    plt.figure(figsize=(10,5)); plt.bar([r["model"]+"-"+r["aspect"] for r in rows],[r["fmax"] for r in rows]); plt.xticks(rotation=45,ha="right"); plt.ylabel("F-max"); plt.tight_layout(); plt.savefig(out/"results/fmax_by_model_aspect.png",dpi=160)
    embedding_config=json.loads((prep/"embedding_config.json").read_text()) if (prep/"embedding_config.json").exists() else {"model_name":"facebook/esm2_t6_8M_UR50D"}
    cfg={"embedding_model":embedding_config["model_name"],"embedding_dim":int(X.shape[1]),"hidden_dim":512,"seed":seed}
    export_model(out/"models",hier,term_lists,name_map,pairs,cfg)
    torch.save(flat.cpu().state_dict(),out/"models/flat.pt")
    print("Wrote",out/"models","and",out/"results")

if __name__=="__main__": main()
