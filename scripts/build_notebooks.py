import json
from pathlib import Path
def cell(kind, text):
    d={"cell_type":kind,"metadata":{},"source":text.splitlines(True)}
    if kind=="code": d.update(execution_count=None,outputs=[])
    return d
config='''from pathlib import Path\nimport random,json,subprocess,shutil\nimport numpy as np\nSEED=42\nDATA_DIR = "/kaggle/input/competitions/cafa-5-protein-function-prediction"\nFASTA=f"{DATA_DIR}/Train/train_sequences.fasta"\nTERMS=f"{DATA_DIR}/Train/train_terms.tsv"\nOBO=f"{DATA_DIR}/Train/go-basic.obo"\nTAXONOMY=f"{DATA_DIR}/Train/train_taxonomy.tsv"\nIA_FILE=f"{DATA_DIR}/IA.txt"\nWORK=Path("/kaggle/working/protfunc"); WORK.mkdir(parents=True,exist_ok=True)\nrandom.seed(SEED); np.random.seed(SEED)\nmissing=[p for p in [FASTA,TERMS,OBO,TAXONOMY,IA_FILE] if not Path(p).is_file()]\nif missing: raise FileNotFoundError("Missing CAFA inputs: "+", ".join(missing))\nprint("All CAFA input files exist")'''
config=config.replace("missing=[p for p in [FASTA,TERMS,OBO,TAXONOMY,IA_FILE] if not Path(p).is_file()]", "missing=[p for p in [FASTA,TERMS,OBO,TAXONOMY] if not Path(p).is_file()]")
config=config.replace('print("All CAFA input files exist")', 'if not Path(IA_FILE).is_file(): print("Optional IA.txt unavailable; S-min is outside this reduced scope.")\nprint("All required CAFA input files exist")')
config=config.replace("SEED=42"+chr(10)+"DATA_DIR", "SEED=42"+chr(10)+"TOP_K=200"+chr(10)+"DATA_DIR")
config=config.replace("TOP_K=200"+chr(10)+"DATA_DIR", "TOP_K=200"+chr(10)+"MODEL_NAME=\"facebook/esm2_t6_8M_UR50D\""+chr(10)+"DATA_DIR")
config=config.replace('WORK=Path("/kaggle/working/protfunc"); WORK.mkdir(parents=True,exist_ok=True)', 'WORK=Path("/kaggle/working/protfunc")')
config=config.replace('print("All required CAFA input files exist")', 'WORK.mkdir(parents=True,exist_ok=True)\nprint("All required CAFA input files exist")')
prep=[cell("markdown","# 01 Prepare and embed\n\nKaggle GPU notebook. Estimated runtime: 10-30 min. Run cells in order. Outputs live in `/kaggle/working/protfunc`; embedding shards can be resumed."),cell("code",config),cell("markdown","## Install dependencies and check MMseqs2"),cell("code",'''!pip -q install biopython\nMMSEQS_OK=shutil.which("mmseqs") is not None\nif not MMSEQS_OK:\n try:\n  !apt-get update -qq && apt-get install -y -qq mmseqs2\n  MMSEQS_OK=shutil.which("mmseqs") is not None\n except Exception as e: print("Random split fallback:",e)\nprint("MMseqs2:",MMSEQS_OK)'''),cell("markdown","## Subsample, propagate labels, choose top 200 per aspect"),cell("code",'''from Bio import SeqIO\nfrom collections import defaultdict,Counter\nseqs={r.id:str(r.seq).upper() for r in SeqIO.parse(FASTA,"fasta") if len(r.seq)<=512}\nids=sorted(seqs); random.shuffle(ids); ids=ids[:40000]; seqs={k:seqs[k] for k in ids}\nann=defaultdict(set)\nfor row in open(TERMS):\n c=row.rstrip().split("\\t")\n if len(c)>=2 and c[0] in seqs: ann[c[0]].add(c[1])\nparents=defaultdict(set); aspect={}; term=None; ns={"molecular_function":"MF","biological_process":"BP","cellular_component":"CC"}\nfor raw in open(OBO):\n line=raw.strip()\n if line=="[Term]": term=None\n elif line.startswith("id: GO:"): term=line[4:]\n elif term and line.startswith("is_a: GO:"): parents[term].add(line.split()[1])\n elif term and line.startswith("namespace:"): aspect[term]=ns.get(line.split(":",1)[1].strip())\nlabels={}\nfor p,ts in ann.items():\n x=set(ts); todo=list(ts)\n while todo:\n  for q in parents[todo.pop()]:\n   if q not in x: x.add(q); todo.append(q)\n labels[p]=x\ncounts=Counter(t for ts in labels.values() for t in ts)\nterm_lists={a:[t for t,n in sorted(((t,n) for t,n in counts.items() if aspect.get(t)==a),key=lambda z:(-z[1],z[0]))[:200] for a in ("MF","BP","CC")}\nprint(len(seqs),{a:len(v) for a,v in term_lists.items()})'''),cell("markdown","## Cluster at 30% identity; assign whole clusters; record fallback"),cell("code",'''from Bio.SeqRecord import SeqRecord\nfrom Bio.Seq import Seq\nfasta=WORK/"subset.fasta"; SeqIO.write([SeqRecord(Seq(v),id=k,description="") for k,v in seqs.items()],fasta,"fasta")\nmeta={"seed":SEED,"method":"random_fallback"}; groups={}\nif MMSEQS_OK:\n try:\n  prefix=str(WORK/"cluster"); subprocess.run(["mmseqs","easy-cluster",str(fasta),prefix,str(WORK/"tmp"),"--min-seq-id","0.3","-c","0.8"],check=True)\n  groups=defaultdict(list)\n  for row in open(prefix+"_cluster.tsv"):\n   rep,member=row.rstrip().split("\\t"); groups[rep].append(member)\n  meta["method"]="mmseqs_easy_cluster"\n except Exception as e: print("MMseqs failed; random fallback:",e)\nif not groups: groups={k:[k] for k in seqs}\ngs=list(groups.values()); random.Random(SEED).shuffle(gs); n=sum(map(len,gs)); seen=0; splits={"train":[],"val":[],"test":[]}\nfor g in gs:\n s="train" if seen<.8*n else "val" if seen<.9*n else "test"; splits[s]+=g; seen+=len(g)\nmeta["sizes"]={k:len(v) for k,v in splits.items()}; json.dump({"splits":splits,"metadata":meta},open(WORK/"splits.json","w"),indent=2)'''),cell("markdown","## Extract frozen ESM-2 mean pooled embeddings in resumable shards"),cell("code",'''import torch\nfrom transformers import AutoTokenizer,AutoModel\nMODEL_NAME="facebook/esm2_t6_8M_UR50D"; assert torch.cuda.is_available(),"Enable Kaggle GPU"\ntok=AutoTokenizer.from_pretrained(MODEL_NAME); model=AutoModel.from_pretrained(MODEL_NAME).cuda().eval(); ed=WORK/"embeddings"; ed.mkdir(exist_ok=True); all_ids=list(seqs)\nfor lo in range(0,len(all_ids),1000):\n out=ed/f"emb_{lo:06d}.npy"; idfile=ed/f"ids_{lo:06d}.json"\n if out.exists() and idfile.exists(): continue\n keys=all_ids[lo:lo+1000]; chunks=[]\n with torch.no_grad():\n  for b in range(0,len(keys),16):\n   batch=tok([seqs[k] for k in keys[b:b+16]],return_tensors="pt",padding=True,truncation=True,max_length=514).to("cuda"); h=model(**batch).last_hidden_state; mask=batch["attention_mask"].unsqueeze(-1); chunks.append(((h*mask).sum(1)/mask.sum(1).clamp(min=1)).cpu().numpy().astype(np.float16))\n np.save(out,np.concatenate(chunks)); json.dump(keys,open(idfile,"w"))\nprint("Shards saved to",ed)''')]
train=[cell("markdown","Notebook 02"),cell("code",config),cell("markdown","Verify inputs"),cell("code",""),cell("markdown","Train and export"),cell("code",""),cell("markdown","Optional DIAMOND"),cell("code","")]
prep[0]=cell("markdown","# 01 Prepare and embed\n\nKaggle GPU notebook. Estimated runtime: 10-30 min. Run cells in order. Outputs are written under `/kaggle/working/protfunc`; completed embedding shards are skipped on rerun.")
prep[3]=cell("code",'''!pip -q install biopython
MMSEQS_OK=shutil.which("mmseqs") is not None
if not MMSEQS_OK:
    try:
        subprocess.run(["apt-get","update","-qq"],check=True)
        subprocess.run(["apt-get","install","-y","-qq","mmseqs2"],check=True)
        MMSEQS_OK=shutil.which("mmseqs") is not None
    except Exception as e: print("MMseqs2 install failed; random split fallback:",e)
print("MMseqs2 available:",MMSEQS_OK)''')
train[0]=cell("markdown","# 02 Train, evaluate, export\n\nKaggle GPU notebook. Estimated runtime: 5-20 min. Attach notebook 01 outputs and this repository as Kaggle inputs.")
train[1]=cell("code",config+'''\nimport glob
source_hits=glob.glob("/kaggle/input/**/src/protfunc/models.py",recursive=True)
if not source_hits: raise FileNotFoundError("Could not find src/protfunc/models.py under /kaggle/input")
SOURCE=Path(source_hits[0]).parents[2]
split_hits=sorted(glob.glob("/kaggle/input/**/splits.json",recursive=True)+glob.glob("/kaggle/working/**/splits.json",recursive=True))
if not split_hits: raise FileNotFoundError("Could not find splits.json under /kaggle/input or /kaggle/working")
PREP=Path(split_hits[0]).parent
assert (PREP/"embeddings").is_dir(),f"Prepared embeddings directory missing under {PREP}"''')
train[3]=cell("code",'''subprocess.run(["python","-m","pip","install","-q","-r",str(SOURCE/"requirements-train.txt")],check=True)
assert all(Path(p).is_file() for p in [FASTA,TERMS,OBO,TAXONOMY]),"Required CAFA input missing"
assert (PREP/"splits.json").is_file() and list((PREP/"embeddings").glob("emb_*.npy")),"Prepared split or embedding shards missing"
print("All source, CAFA, and prepared files exist")''')
for filename,cells in [("01_prepare_and_embed.ipynb",prep),("02_train_evaluate_export.ipynb",train)]:
    if filename.startswith("01"):
        cells[5]["source"]=[x.replace('[:200] for a in ("MF","BP","CC")}', '[:200]] for a in ("MF","BP","CC")}').replace('[:200]', '[:TOP_K]').replace('if aspect.get(t)==a', 'if aspect.get(t)==a and t not in {"GO:0003674","GO:0008150","GO:0005575"}') for x in cells[5]["source"]]
        cells[5]["source"].extend(['\n','json.dump(term_lists,open(WORK/"term_lists.json","w"),indent=2)\n','json.dump({p:sorted(ts) for p,ts in labels.items()},open(WORK/"labels.json","w"),indent=2)\n'])
        cells[9]["source"]=[x.replace('MODEL_NAME="facebook/esm2_t6_8M_UR50D"; assert torch.cuda.is_available()', 'assert torch.cuda.is_available()') for x in cells[9]["source"]]
        cells[9]["source"].extend(['\n','json.dump({"model_name":MODEL_NAME,"seed":SEED},open(WORK/"embedding_config.json","w"),indent=2)\n'])
    else:
        cells[5]=cell("code",'''# SOURCE and PREP were found by glob in the setup cell.
import sys,runpy
sys.path.insert(0,str(SOURCE/"src")); sys.path.insert(0,str(SOURCE/"notebooks"))
sys.argv=["train_gpu.py","--data-dir",DATA_DIR,"--prep-dir",str(PREP),"--output-dir","/kaggle/working"]
runpy.run_path(str(SOURCE/"notebooks/train_gpu.py"),run_name="__main__")''')
        cells[7]=cell("code",'''# Optional: run only if time permits. Database is built from the training split only.
import subprocess,sys,runpy
subprocess.run(["apt-get","update","-qq"],check=True)
subprocess.run(["apt-get","install","-y","-qq","diamond-aligner"],check=True)
sys.argv=["run_diamond.py",DATA_DIR,str(PREP),"/kaggle/working"]
runpy.run_path(str(SOURCE/"notebooks/run_diamond.py"),run_name="__main__")''')
        cells[6]=cell("markdown","## Optional DIAMOND top-hit baseline\nThis uses training sequences only as the database, transfers top-hit annotations weighted by bitscore, and appends measured metrics. Skip this section if time is short.")
        cells=cells[:8]
    for item in cells:
        if item["cell_type"]=="code":
            source="".join(item["source"])
            source=source.replace('  !apt-get update -qq && apt-get install -y -qq mmseqs2','  subprocess.run(["apt-get","update","-qq"],check=True); subprocess.run(["apt-get","install","-y","-qq","mmseqs2"],check=True)')
            source=source.replace('[FASTA,TERMS,OBO,TAXONOMY,IA_FILE]','[FASTA,TERMS,OBO,TAXONOMY]')
            source=source.replace('return_tensors="pt",padding=True,truncation=True,max_length=514).to("cuda"); h=model(**batch).last_hidden_state; mask=batch["attention_mask"].unsqueeze(-1);','return_tensors="pt",padding=True,truncation=True,max_length=514,return_special_tokens_mask=True).to("cuda"); special=batch.pop("special_tokens_mask"); h=model(**batch).last_hidden_state; mask=(batch["attention_mask"]*(1-special)).unsqueeze(-1);')
            item["source"]=source.splitlines(True)
    obj={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}},"nbformat":4,"nbformat_minor":5}
    Path("notebooks",filename).write_text(json.dumps(obj,indent=1),encoding="utf-8")
