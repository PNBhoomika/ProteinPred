"""Evaluation helpers for GO multi-label predictions."""
import numpy as np

METRIC_COLUMNS=("model","aspect","fmax","micro_fmax","mean_term_ap","rare_fmax","frequent_fmax","split_method")

def micro_fmax(y_true,y_score):
    y=np.asarray(y_true,dtype=bool); s=np.asarray(y_score,dtype=float); best=0.0
    for threshold in np.linspace(0,1,101):
        pred=s>=threshold; tp=np.logical_and(pred,y).sum(); fp=np.logical_and(pred,~y).sum(); fn=np.logical_and(~pred,y).sum()
        precision=tp/(tp+fp) if tp+fp else 0.; recall=tp/(tp+fn) if tp+fn else 0.
        if precision+recall: best=max(best,2*precision*recall/(precision+recall))
    return float(best)

def fmax(y_true,y_score):
    """CAFA-style protein-centric F-max over per-protein precision and recall."""
    y=np.asarray(y_true,dtype=bool); s=np.asarray(y_score,dtype=float)
    if y.ndim==1: y=y[None,:]; s=s[None,:]
    true_counts=y.sum(axis=1); recall_eligible=true_counts>0
    if not np.any(recall_eligible): return 0.0
    best=0.0
    for threshold in np.linspace(0,1,101):
        pred=s>=threshold; pred_counts=pred.sum(axis=1); true_positive=np.logical_and(pred,y).sum(axis=1)
        precision_eligible=pred_counts>0
        precision=float(np.mean(true_positive[precision_eligible]/pred_counts[precision_eligible])) if np.any(precision_eligible) else 0.0
        recall=float(np.mean(true_positive[recall_eligible]/true_counts[recall_eligible]))
        if precision+recall: best=max(best,2*precision*recall/(precision+recall))
    return float(best)

def mean_term_ap(y_true,y_score):
    y=np.asarray(y_true,dtype=bool); s=np.asarray(y_score,dtype=float)
    if y.ndim==1: y=y[:,None]; s=s[:,None]
    aps=[]
    for col in range(y.shape[1]):
        order=np.argsort(-s[:,col],kind="stable"); ordered_scores=s[order,col]; truth=y[order,col]; positives=int(truth.sum())
        if not positives: aps.append(0.0); continue
        cumulative=np.cumsum(truth); ends=np.r_[np.flatnonzero(np.diff(ordered_scores)),len(truth)-1]
        precision=cumulative[ends]/(ends+1); recall=cumulative[ends]/positives
        aps.append(float(np.sum(np.diff(np.r_[0.0,recall])*precision)))
    return float(np.mean(aps)) if aps else 0.0

def hierarchy_enforce(scores,parent_child):
    scores=np.asarray(scores).copy()
    parents={}; children={}; edges=np.asarray(parent_child,dtype=int).reshape(-1,2)
    for parent,child in edges:
        parents.setdefault(int(child),[]).append(int(parent)); children.setdefault(int(parent),[]).append(int(child))
    indegree={i:len(parents.get(i,())) for i in range(scores.shape[-1])}; queue=[i for i,d in indegree.items() if d==0]; depth={i:0 for i in queue}; cursor=0
    while cursor<len(queue):
        parent=queue[cursor]; cursor+=1
        for child in children.get(parent,()):
            depth[child]=max(depth.get(child,0),depth[parent]+1); indegree[child]-=1
            if indegree[child]==0: queue.append(child)
    for child,ps in sorted(parents.items(),key=lambda item:depth.get(item[0],0),reverse=True):
        unique=np.asarray(sorted(set(ps)),dtype=int)
        scores[...,unique]=np.maximum(scores[...,unique],scores[...,child,None])
    return scores

def aspect_metrics(y_true,y_score,term_frequencies):
    freq=np.asarray(term_frequencies); order=np.argsort(freq); n=max(1,len(order)//3)
    y=np.asarray(y_true); s=np.asarray(y_score)
    return {"fmax":fmax(y,s),"micro_fmax":micro_fmax(y,s),"mean_term_ap":mean_term_ap(y,s),"rare_fmax":fmax(y[:,order[:n]],s[:,order[:n]]),"frequent_fmax":fmax(y[:,order[-n:]],s[:,order[-n:]])}
