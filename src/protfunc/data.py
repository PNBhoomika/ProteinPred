from collections import Counter, defaultdict
import random

ROOT_TERMS = {"GO:0003674", "GO:0008150", "GO:0005575"}

def exclude_root_terms(term_lists):
    """Return aspect term lists with all three GO ontology roots removed."""
    return {aspect:[term for term in terms if term not in ROOT_TERMS] for aspect,terms in term_lists.items()}

def exclude_root_labels(annotations, parents):
    """Remove GO roots from direct annotations and parent edges before propagation."""
    clean_annotations={protein:set(terms)-ROOT_TERMS for protein,terms in annotations.items()}
    clean_parents={term:set(term_parents)-ROOT_TERMS for term,term_parents in parents.items()}
    return clean_annotations,clean_parents

def parse_fasta(path):
    out={}; key=None
    with open(path,encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line.startswith(">"):
                key=line[1:].split()[0]; out[key]=""
            elif line and key: out[key]+=line.upper()
    return out

def parse_terms(path):
    out=defaultdict(set)
    with open(path,encoding="utf-8") as f:
        for i,line in enumerate(f):
            c=line.rstrip("\n").split("\t")
            if i==0 and c[0].lower() in {"protein","protein_id"}: continue
            if len(c)>=2: out[c[0]].add(c[1])
    return dict(out)

def parse_obo(path):
    parents=defaultdict(set); aspects={}; obsolete=set(); term=None
    names={"molecular_function":"MF","biological_process":"BP","cellular_component":"CC"}
    with open(path,encoding="utf-8") as f:
        for raw in f:
            line=raw.strip()
            if line=="[Term]": term=None
            elif line.startswith("id: GO:"): term=line[4:]
            elif term and line.startswith("is_a: GO:"): parents[term].add(line.split()[1])
            elif term and line.startswith("namespace:"): aspects[term]=names.get(line.split(":",1)[1].strip())
            elif term and line=="is_obsolete: true": obsolete.add(term)
    for t in obsolete: parents.pop(t,None); aspects.pop(t,None)
    return dict(parents),aspects

def propagate_labels(annotations,parents):
    result={}
    for protein,terms in annotations.items():
        expanded=set(terms); todo=list(terms)
        while todo:
            for p in parents.get(todo.pop(),()):
                if p not in expanded: expanded.add(p); todo.append(p)
        result[protein]=expanded
    return result

def select_terms(labels,aspects,top_k=200):
    counts=Counter(t for ts in labels.values() for t in ts)
    return {a:[t for t,_ in sorted(((t,n) for t,n in counts.items() if aspects.get(t)==a and t not in ROOT_TERMS),key=lambda x:(-x[1],x[0]))[:top_k]] for a in ("MF","BP","CC")}

def random_subset(sequences,n=40000,max_length=512,seed=42):
    ids=sorted(k for k,v in sequences.items() if len(v)<=max_length); random.Random(seed).shuffle(ids)
    return {k:sequences[k] for k in ids[:n]}

def assign_cluster_splits(cluster_members,seed=42,ratios=(.8,.1,.1)):
    groups=list(cluster_members); random.Random(seed).shuffle(groups); total=sum(map(len,groups)); a,b=ratios[0]*total,(ratios[0]+ratios[1])*total
    out={"train":[],"val":[],"test":[]}; seen=0
    for group in groups:
        split="train" if seen<a else "val" if seen<b else "test"; out[split].extend(group); seen+=len(group)
    return out

def load_splits(path):
    import json
    with open(path,encoding="utf-8") as f: return json.load(f)
