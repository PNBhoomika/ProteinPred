import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from protfunc.eval import fmax, micro_fmax, hierarchy_enforce

def test_fmax_hand_example():
    assert fmax(np.array([[1,0],[0,1]]),np.array([[.9,.1],[.1,.8]]))==1.0

def test_protein_centric_fmax_differs_from_pooled_fmax():
    y=np.zeros((2,10),dtype=int); y[0,0]=1; y[1,:9]=1
    scores=np.zeros((2,10),dtype=float); scores[0,0]=.8; scores[0,1]=.9; scores[1,:9]=.9
    assert np.isclose(fmax(y,scores),6/7)
    assert np.isclose(micro_fmax(y,scores),20/21)

def test_hierarchy_scores_parent_at_least_child():
    scores=hierarchy_enforce(np.array([[.2,.9,.1]]),[(0,1),(1,2)])
    assert scores[0,0]>=scores[0,1]>=scores[0,2]

def test_hierarchy_model_output_constraint():
    # Model (b)'s max-constraint layer uses the same hierarchy projection.
    logits=np.array([[0.0,3.0,-1.0]])
    scores=1/(1+np.exp(-logits))
    constrained=hierarchy_enforce(scores,[(0,1),(1,2)])
    assert constrained[0,0]>=constrained[0,1]>=constrained[0,2]

def test_model_b_hierarchy_consistency():
    import pytest
    torch=pytest.importorskip("torch")
    from protfunc.models import MultiHeadMLP
    model=MultiHeadMLP(4,{"MF":["p","c"],"BP":[],"CC":[]},hidden_dim=8,hierarchy=True,parent_child=[(0,1)])
    scores=model(torch.randn(5,4))
    assert torch.all(scores[:,0]>=scores[:,1])

def test_api_mock_embedder(monkeypatch):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    import api.main as main
    class FakeModel:
        def __call__(self,x): return np.array([[0.0,1.0]])
    monkeypatch.setattr(main,"demo",False)
    monkeypatch.setattr(main,"model",FakeModel())
    monkeypatch.setattr(main,"terms",{"MF":["GO:1","GO:2"],"BP":[],"CC":[]})
    monkeypatch.setattr(main,"names",{"GO:1":"one","GO:2":"two"})
    monkeypatch.setattr(main,"embedder",lambda protein: np.array([0.1,0.2]))
    response=main.predict(main.PredictRequest(dna="ATGAAATAA"))
    assert response["protein"]=="MK" and not response["demo_model"]
    assert response["predictions"]["MF"][0]["go_id"]=="GO:2"

def test_api_no_orf_returns_clear_error():
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    import pytest
    from fastapi import HTTPException
    import api.main as main
    with pytest.raises(HTTPException) as exc:
        main.predict(main.PredictRequest(dna="CCCCCC"))
    assert exc.value.status_code==422
    assert "No ORF found" in exc.value.detail

