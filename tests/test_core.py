import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import pytest
from protfunc.dna import clean_dna, find_orfs, longest_orf
from protfunc.data import propagate_labels, assign_cluster_splits, select_terms

def test_dna_header_whitespace_and_orf():
    assert clean_dna(">x\nATG AAA TAA") == "ATGAAATAA"
    best, rest=longest_orf("ATGAAATAA")
    assert best.protein=="MK" and best.complete and rest==[]

def test_orf_reverse_frame_and_no_orf():
    # Reverse complement contains ATGAAATAA.
    assert any(x.frame.startswith("-") and x.protein=="MK" for x in find_orfs("TTATTTCAT"))
    with pytest.raises(ValueError,match="No ORF"): longest_orf("CCCCCC")

def test_invalid_dna():
    with pytest.raises(ValueError,match="Invalid"): clean_dna("ATGX")
    with pytest.raises(ValueError,match="empty"): clean_dna("  ")

def test_propagation():
    assert propagate_labels({"p":{"c"}},{"c":{"p"}})["p"]=={"c","p"}

def test_term_selection_excludes_ontology_roots():
    roots={"GO:0003674":"MF","GO:0008150":"BP","GO:0005575":"CC"}
    aspects={**roots,"GO:0000001":"MF","GO:0000002":"BP","GO:0000003":"CC"}
    terms=select_terms({"p":set(aspects)},aspects,top_k=10)
    assert all(root not in terms[aspect] for root,aspect in roots.items())
    assert terms=={"MF":["GO:0000001"],"BP":["GO:0000002"],"CC":["GO:0000003"]}

def test_whole_cluster_split_no_leakage():
    clusters=[["a","b"],["c"],["d","e"],["f"],["g"],["h"]]
    splits=assign_cluster_splits(clusters,seed=1)
    membership={x:s for s,ids in splits.items() for x in ids}
    for group in clusters: assert len({membership[x] for x in group})==1
