"""FastAPI service for DNA-to-GO predictions (demo mode without trained artifacts)."""
from pathlib import Path
import json
import sys
import hashlib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from protfunc.dna import clean_dna,longest_orf

class PredictRequest(BaseModel):
    dna: str = Field(min_length=3,max_length=1000000)

class GOTerm(BaseModel):
    go_id: str
    name: str
    score: float

class PredictResponse(BaseModel):
    orf: dict
    protein: str
    predictions: dict[str,list[GOTerm]]
    demo_model: bool
    note: str

app=FastAPI(title="ProtFunc",version="0.1.0")
MODEL_DIR=ROOT/"models"
ARTIFACT=MODEL_DIR/"config.json"
demo=True; config={}; terms={}; names={}; model=None; embedder=None

def _load_artifacts():
    global demo,config,terms,names,model
    if not ARTIFACT.exists(): return
    try:
        import numpy as np
        import torch
        from protfunc.models import MultiHeadMLP
        config=json.loads(ARTIFACT.read_text(encoding="utf-8")); terms=json.loads((MODEL_DIR/"terms.json").read_text(encoding="utf-8")); names=json.loads((MODEL_DIR/"names.json").read_text(encoding="utf-8"))
        pairs=np.load(MODEL_DIR/"parent_child.npy"); model=MultiHeadMLP(config["embedding_dim"],terms,config.get("hidden_dim",512),hierarchy=True,parent_child=pairs)
        model.load_state_dict(torch.load(MODEL_DIR/"hierarchy.pt",map_location="cpu",weights_only=True)); model.eval(); demo=False
    except Exception as exc:
        # Incomplete/mismatched exports should remain clearly labeled demo output.
        print(f"Could not load trained ProtFunc artifact; staying in demo mode: {exc}")

_load_artifacts()

def set_embedder(fn):
    """Injectable for API tests and embedding cache integrations."""
    global embedder; embedder=fn

def _predict(protein):
    if demo:
        # Clearly illustrative deterministic scores, never training metrics.
        digest=hashlib.sha256(protein.encode()).digest(); out={}
        for ai,(aspect,examples) in enumerate({"MF":[("GO:0005515","protein binding"),("GO:0003674","molecular function")],"BP":[("GO:0009987","cellular process"),("GO:0008150","biological process")],"CC":[("GO:0005737","cytoplasm"),("GO:0005623","cellular component")]} .items()):
            vals=[]
            for i,(go,name) in enumerate(examples): vals.append({"go_id":go,"name":name,"score":round(.5+digest[(ai*3+i)%len(digest)]/255*.45,3)})
            out[aspect]=sorted(vals,key=lambda x:x["score"],reverse=True)
        return out
    if embedder is None:
        from protfunc.embed import embed_one
        vector=embed_one(protein,config.get("embedding_model","facebook/esm2_t6_8M_UR50D"))
    else: vector=embedder(protein)
    try:
        import torch
    except ImportError:
        output=model(np.asarray(vector,dtype=np.float32).reshape(1,-1))
    else:
        with torch.no_grad(): output=model(torch.as_tensor(vector,dtype=torch.float32).reshape(1,-1))
    logits=output.detach().cpu().numpy().reshape(-1) if hasattr(output,"detach") else np.asarray(output).reshape(-1)
    scores=1/(1+np.exp(-logits))
    flat=terms if isinstance(terms,list) else sum((terms[a] for a in ("MF","BP","CC")),[])
    ranked={a:[] for a in ("MF","BP","CC")}
    for i,go in enumerate(flat):
        a=next((a for a,ts in (terms.items() if isinstance(terms,dict) else []) if go in ts),None)
        if a: ranked[a].append({"go_id":go,"name":names.get(go,go),"score":float(scores[i])})
    return {a:sorted(v,key=lambda x:x["score"],reverse=True)[:10] for a,v in ranked.items()}

@app.get("/health")
def health(): return {"status":"ok","demo_model":demo}

@app.post("/predict",response_model=PredictResponse)
def predict(payload:PredictRequest):
    try:
        dna=clean_dna(payload.dna,max_length=1000000); best,others=longest_orf(dna,min_aa=1)
    except ValueError as exc: raise HTTPException(status_code=422,detail=str(exc)) from exc
    return {"orf":{"frame":best.frame,"start":best.start,"end":best.end,"complete":best.complete,"other_candidates":len(others)},"protein":best.protein,"predictions":_predict(best.protein),"demo_model":demo,"note":"Computational hypotheses only; predictions require experimental validation."}

@app.get("/")
def index(): return FileResponse(ROOT/"web"/"index.html")
