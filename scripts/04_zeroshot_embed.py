"""Zero-shot EpiAgent [CLS] cell embeddings for every GSE194122 multiome BMMC cell.

1. global_TFIDF + tokenization (vectorised equivalent of epiagent.tokenization, checked against it).
   TF-IDF is per-cell (global document frequencies), so the saved sentences are reusable for any split.
2. Pretrained EpiAgent forward pass; embedding = transformer output at position 0 ([CLS]), exactly as
   epiagent.inference.infer_cell_embeddings. Sentences are truncated like CellDataset(is_random=False):
   top max_length-2 cCREs by TF-IDF, then [CLS] ... [SEP]. Cells are batched in length order to cut
   padding (padding is masked via key_padding_mask) and restored to original order afterwards.
"""
import json
import os
import time
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
import torch
from torch.utils.data import DataLoader, Dataset

from epiagent.dataset import collate_fn
from epiagent.model import EpiAgent
from epiagent.preprocessing import global_TFIDF
from epiagent.tokenization import tokenization

ROOT = Path(__file__).resolve().parents[1]
CCRE_PATH = f"{ROOT}/data/GSE194122/GSE194122_multiome_BMMC_cell_by_cCRE.h5ad"
DF_PATH = f"{ROOT}/EpiAgent/data/cCRE_document_frequency.npy"
MODEL_PATH = f"{ROOT}/models/pretrained_EpiAgent.pth"
SENTENCES_PATH = f"{ROOT}/data/GSE194122/cell_sentences.npz"
OUT_DIR = f"{ROOT}/results/zeroshot"
MAX_LENGTH = 8192
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 32))

os.makedirs(OUT_DIR, exist_ok=True)
t0 = time.time()

# 1. TF-IDF and tokenization.
adata = sc.read_h5ad(CCRE_PATH)
raw_nnz = np.diff(adata.X.indptr)
adata = global_TFIDF(adata, np.load(DF_PATH))
X = adata.X.tocsr()
X.sort_indices()
assert (np.diff(X.indptr) == raw_nnz).all(), "TF-IDF changed the sparsity pattern"

indptr = X.indptr.astype(np.int64)
tokens = np.empty(X.nnz, dtype=np.int32)
for i in range(X.shape[0]):
    s, e = indptr[i], indptr[i + 1]
    tokens[s:e] = X.indices[s:e][np.argsort(-X.data[s:e])] + 4
print(f"tokenized {X.shape[0]:,} cells, {X.nnz:,} tokens [{time.time() - t0:.0f}s]")

# Check against the library implementation on a subset.
check = adata[:200].copy()
tokenization(check)
for i, sent in enumerate(check.obs["cell_sentences"]):
    assert json.loads(sent) == tokens[indptr[i]:indptr[i + 1]].tolist(), f"mismatch at cell {i}"
print("tokens match epiagent.tokenization on 200 cells")

np.savez(SENTENCES_PATH, tokens=tokens, indptr=indptr, obs_names=adata.obs_names.values.astype(str))
print(f"saved {SENTENCES_PATH}")
obs = adata.obs.copy()
del adata, X, check


# 2. Embeddings.
class SentenceDataset(Dataset):
    """CellDataset(is_random=False).__getitem__ over flat token arrays, visiting cells in `order`."""

    def __init__(self, tokens, indptr, order, max_length=MAX_LENGTH):
        self.tokens, self.indptr, self.order, self.max_length = tokens, indptr, order, max_length

    def __len__(self):
        return len(self.order)

    def __getitem__(self, k):
        i = self.order[k]
        cell = self.tokens[self.indptr[i]:self.indptr[i + 1]][:self.max_length - 2]
        return torch.tensor(np.concatenate(([1], cell, [2])), dtype=torch.long)


lengths = np.diff(indptr)
order = np.argsort(-lengths, kind="stable")
loader = DataLoader(SentenceDataset(tokens, indptr, order), batch_size=BATCH_SIZE, shuffle=False,
                    num_workers=8, collate_fn=collate_fn)

device = torch.device("cuda:0")
print(f"device: {torch.cuda.get_device_name(0)}, batch size {BATCH_SIZE}")
model = EpiAgent(vocab_size=1355449, num_layers=18, embedding_dim=512, num_attention_heads=8,
                 max_rank_embeddings=8192, use_flash_attn=True,
                 pos_weight_for_RLM=torch.tensor(1.), pos_weight_for_CCA=torch.tensor(1.))
print(model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu")))
model.to(device)
model.eval()

sorted_emb = np.empty((len(order), 512), dtype=np.float32)
done = 0
t1 = time.time()
for b, batch in enumerate(loader):
    input_ids = batch.to(device)
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            output = model(input_ids, return_transformer_output=True)
            sorted_emb[done:done + len(batch)] = output["transformer_outputs"][:, 0, :].float().cpu().numpy()
    done += len(batch)
    if b % 200 == 0 or done == len(order):
        print(f"  {done:,}/{len(order):,} cells [{time.time() - t1:.0f}s]", flush=True)

emb = np.empty_like(sorted_emb)
emb[order] = sorted_emb
assert np.isfinite(emb).all(), "non-finite embeddings"

out = ad.AnnData(obs=obs, obsm={"X_epiagent": emb})
out.obs["sentence_length"] = lengths
out.write(f"{OUT_DIR}/epiagent_zeroshot_embeddings.h5ad")
print(f"saved {OUT_DIR}/epiagent_zeroshot_embeddings.h5ad [total {time.time() - t0:.0f}s]")
