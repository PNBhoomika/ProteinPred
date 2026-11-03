"""ESM-2 embeddings. Bulk extraction belongs in the GPU notebooks only."""
from functools import lru_cache
import numpy as np

DEFAULT_MODEL = "facebook/esm2_t6_8M_UR50D"

@lru_cache(maxsize=2)
def _load_esm(model_name):
    import torch
    from transformers import AutoModel, AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(model_name)
    model=AutoModel.from_pretrained(model_name).to("cpu").eval()
    return tokenizer,model

def embed_one(sequence: str, model_name: str = DEFAULT_MODEL):
    """Mean-pool last hidden states for one protein on CPU (for the API)."""
    import torch
    tokenizer,model=_load_esm(model_name)
    batch=tokenizer(sequence,return_tensors="pt",truncation=True,max_length=514,return_special_tokens_mask=True)
    special=batch.pop("special_tokens_mask")
    with torch.no_grad():
        hidden=model(**batch).last_hidden_state
        mask=(batch["attention_mask"]*(1-special)).unsqueeze(-1)
        return ((hidden*mask).sum(1)/mask.sum(1).clamp(min=1)).squeeze(0).numpy()

def embed_batch(sequences, model_name=DEFAULT_MODEL, batch_size=16, device="cuda"):
    """Batch helper for notebook use. Do not run extraction locally."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    if device != "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Bulk embedding extraction is intended for a GPU notebook")
    tokenizer=AutoTokenizer.from_pretrained(model_name); model=AutoModel.from_pretrained(model_name).to(device).eval(); out=[]
    with torch.no_grad():
        for start in range(0,len(sequences),batch_size):
            batch=tokenizer(sequences[start:start+batch_size],return_tensors="pt",padding=True,truncation=True,max_length=514,return_special_tokens_mask=True).to(device)
            special=batch.pop("special_tokens_mask"); h=model(**batch).last_hidden_state; mask=(batch["attention_mask"]*(1-special)).unsqueeze(-1)
            out.append(((h*mask).sum(1)/mask.sum(1).clamp(min=1)).cpu().numpy())
    return np.concatenate(out) if out else np.empty((0,model.config.hidden_size),dtype=np.float32)
