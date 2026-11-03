"""Flat and hierarchy-constrained multi-head MLP classifiers."""
import json
from pathlib import Path
import numpy as np

try:
    import torch
    from torch import nn
except ImportError:  # Demo-only installations need not have torch.
    torch=None; nn=None

if nn is not None:
    class MultiHeadMLP(nn.Module):
        def __init__(self,input_dim,term_lists,hidden_dim=512,hierarchy=False,parent_child=None):
            super().__init__(); self.hierarchy=hierarchy
            self.trunk=nn.Sequential(nn.Linear(input_dim,hidden_dim),nn.ReLU(),nn.Dropout(.2),nn.Linear(hidden_dim,hidden_dim//2),nn.ReLU())
            self.aspect_names=("MF","BP","CC")
            self.heads=nn.ModuleDict({a:nn.Linear(hidden_dim//2,len(term_lists[a])) for a in self.aspect_names})
            pairs=np.asarray(parent_child if parent_child is not None else [],dtype=np.int64).reshape(-1,2)
            self.register_buffer("parent_child",torch.as_tensor(pairs,dtype=torch.long),persistent=False)
            parents={}
            children={}
            for parent,child in pairs.tolist():
                parents.setdefault(child,[]).append(parent)
                children.setdefault(parent,[]).append(child)
            indegree={i:len(parents.get(i,())) for i in range(sum(len(term_lists[a]) for a in self.aspect_names))}
            queue=[i for i,d in indegree.items() if d==0]; depth={i:0 for i in queue}; cursor=0
            while cursor<len(queue):
                parent=queue[cursor]; cursor+=1
                for child in children.get(parent,()):
                    depth[child]=max(depth.get(child,0),depth[parent]+1); indegree[child]-=1
                    if indegree[child]==0: queue.append(child)
            by_depth={}
            for child,ps in parents.items():
                for parent in set(ps): by_depth.setdefault(depth.get(child,0),[]).append((parent,child))
            self._propagation=tuple((torch.as_tensor([p for p,c in by_depth[d]],dtype=torch.long),torch.as_tensor([c for p,c in by_depth[d]],dtype=torch.long)) for d in sorted(by_depth,reverse=True))
        def forward(self,x):
            z=torch.cat([self.heads[a](self.trunk(x)) for a in self.aspect_names],dim=1)
            if self.hierarchy and self.parent_child.numel():
                for parents,children in self._propagation:
                    parents=parents.to(z.device); children=children.to(z.device)
                    z=z.scatter_reduce(1,parents.unsqueeze(0).expand(z.shape[0],-1),z.index_select(1,children),reduce="amax",include_self=True)
            return z
else:
    class MultiHeadMLP:
        def __init__(self,*args,**kwargs): raise ImportError("Install the pinned torch dependency to use trained models")

def export_model(output_dir,model,term_lists,names,parent_child,config):
    """Write the artifact layout consumed by api/main.py."""
    if torch is None: raise ImportError("torch is required to export model weights")
    dest=Path(output_dir); dest.mkdir(parents=True,exist_ok=True)
    torch.save(model.state_dict(),dest/("hierarchy.pt" if model.hierarchy else "flat.pt"))
    np.save(dest/"parent_child.npy",np.asarray(parent_child,dtype=np.int64).reshape(-1,2))
    (dest/"terms.json").write_text(json.dumps(term_lists,indent=2),encoding="utf-8")
    (dest/"names.json").write_text(json.dumps(names,indent=2),encoding="utf-8")
    full=dict(config); full["term_lists"]=term_lists; full.setdefault("embedding_model","facebook/esm2_t6_8M_UR50D")
    (dest/"config.json").write_text(json.dumps(full,indent=2),encoding="utf-8")
