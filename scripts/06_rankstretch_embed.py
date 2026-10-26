"""Zero-shot EpiAgent [CLS] embeddings with rank-stretch (extended rank embeddings) for every GSE194122 cell.

Stock EpiAgent (04_zeroshot_embed.py) truncates each cell to its top 8,190 cCREs by TF-IDF, because the rank
table has 8,192 rows. Here the whole sentence is encoded and, for cells longer than the table, position i of a
cell with L tokens (incl. [CLS]/[SEP]) reads rank row floor(i * 8192 / L), as cell_gex's StretchedRankEmbedding.
Cells that fit the table get the identity mapping, so their embeddings should match the truncated run.

Reads data/GSE194122/cell_sentences.npz and results/zeroshot/epiagent_zeroshot_embeddings.h5ad (X_epiagent);
writes results/rankstretch/epiagent_embeddings.h5ad with obsm X_epiagent_truncate (copied) and
X_epiagent_stretch.
"""
import os
import time
from pathlib import Path

import anndata as ad
import numpy as np
import torch
import torch.nn as nn

from epiagent.model import EpiAgent

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = f"{ROOT}/models/pretrained_EpiAgent.pth"
SENTENCES_PATH = f"{ROOT}/data/GSE194122/cell_sentences.npz"
TRUNCATE_PATH = f"{ROOT}/results/zeroshot/epiagent_zeroshot_embeddings.h5ad"
OUT_DIR = f"{ROOT}/results/rankstretch"
RANK_ROWS = 8192
TOKENS_PER_BATCH = int(os.environ.get("TOKENS_PER_BATCH", 200_000))  # padded tokens per forward pass


class StretchedRankEmbedding(nn.Module):
    """Map position i of a cell with L tokens to rank row floor(i * 8192 / max(L, 8192))."""

    def __init__(self, base: nn.Embedding):
        super().__init__()
        self.base = base
        self.lengths = None

    def forward(self, idx):
        size = self.base.num_embeddings
        lengths = self.lengths.clamp(min=size)[:, None]
        return self.base(torch.div(idx * size, lengths, rounding_mode="floor").clamp(max=size - 1))


def batches(lengths):
    """Cell indices grouped longest first, so each batch pads to at most TOKENS_PER_BATCH tokens."""
    batch = []
    for i in np.argsort(-lengths, kind="stable"):
        if batch and (len(batch) + 1) * (lengths[batch[0]] + 2) > TOKENS_PER_BATCH:
            yield batch
            batch = []
        batch.append(i)
    yield batch


os.makedirs(OUT_DIR, exist_ok=True)
t0 = time.time()
s = np.load(SENTENCES_PATH)
tokens, indptr = s["tokens"], s["indptr"]
lengths = np.diff(indptr)
truncated = ad.read_h5ad(TRUNCATE_PATH)
assert (s["obs_names"] == truncated.obs_names.values).all()
over = lengths > RANK_ROWS - 2
print(f"{len(lengths):,} cells, {over.sum():,} ({over.mean():.1%}) over {RANK_ROWS - 2:,} cCREs, "
      f"longest {lengths.max():,}")

device = torch.device("cuda:0")
print(f"device: {torch.cuda.get_device_name(0)}, {TOKENS_PER_BATCH:,} tokens per batch")
model = EpiAgent(vocab_size=1355449, num_layers=18, embedding_dim=512, num_attention_heads=8,
                 max_rank_embeddings=RANK_ROWS, use_flash_attn=True,
                 pos_weight_for_RLM=torch.tensor(1.), pos_weight_for_CCA=torch.tensor(1.))
print(model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu")))
model.rank_embedding = StretchedRankEmbedding(model.rank_embedding)
model.to(device).eval()

emb = np.zeros((len(lengths), 512), dtype=np.float32)
done = 0
t1 = time.time()
with torch.no_grad():
    for b, batch in enumerate(batches(lengths)):
        seqs = [torch.from_numpy(np.concatenate(([1], tokens[indptr[i]:indptr[i + 1]], [2]))).long() for i in batch]
        input_ids = nn.utils.rnn.pad_sequence(seqs, batch_first=True).to(device)
        model.rank_embedding.lengths = (input_ids != 0).sum(dim=1)
        with torch.cuda.amp.autocast():
            out = model(input_ids, return_transformer_output=True)["transformer_outputs"]
        emb[batch] = out[:, 0].float().cpu().numpy()
        done += len(batch)
        if b % 100 == 0 or done == len(lengths):
            print(f"  {done:,}/{len(lengths):,} cells [{time.time() - t1:.0f}s]", flush=True)
assert np.isfinite(emb).all(), "non-finite embeddings"

trunc = truncated.obsm["X_epiagent"]
cos = (emb * trunc).sum(1) / (np.linalg.norm(emb, axis=1) * np.linalg.norm(trunc, axis=1))
print(f"cosine stretch vs truncated: cells <= {RANK_ROWS - 2:,} cCREs min {cos[~over].min():.5f} "
      f"median {np.median(cos[~over]):.5f}; cells over: median {np.median(cos[over]):.4f} min {cos[over].min():.4f}")
assert cos[~over].min() > 0.99, "unstretched cells should match the truncated run"

out = ad.AnnData(obs=truncated.obs[["cell_type", "Site", "DonorID", "batch", "sentence_length"]].copy(),
                 obsm={"X_epiagent_truncate": trunc, "X_epiagent_stretch": emb})
out.write(f"{OUT_DIR}/epiagent_embeddings.h5ad")
print(f"saved {OUT_DIR}/epiagent_embeddings.h5ad [total {time.time() - t0:.0f}s]")
