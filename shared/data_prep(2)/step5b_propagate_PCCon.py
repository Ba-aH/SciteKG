"""
step5b_propagate_PCCon.py
─────────────────────────
Reads:
  feat_C.pt          [N_citations, 768]
  adj_CP_cited.pt    [N_citations, N_papers]  (citation → cited paper)
  node_index.json    (for N_total)
  all_contexts.json  (citing_uri per citation node — for the train mask)
  split_uris.json    (frozen train/val/test citing_uri lists — LEAKAGE GUARD)

LEAKAGE FIX: feat_C.pt already has non-train citation rows zeroed out
(step5a), but row_normalise() previously divided by the FULL row degree
(train+val+test citation count). That both dilutes the mean incorrectly
and leaks the val/test citation *count* through the denominator, even
though the content itself was masked. Fix: mask adj_CP_cited down to only
the citation-node rows belonging to TRAIN citing papers BEFORE
transposing/normalising, so the denominator matches what feat_C actually
contains.

Propagation (identical pattern to step4, now on the train-masked adjacency):
  adj_CP_cited_train        [N_citations, N_papers]  (train rows only, rest zeroed)
  adj_CP_cited_train.T      [N_papers, N_citations]
  row_normalise(adj_CP_cited_train.T) @ feat_C  →  feat_PCCon [N_papers, 768]

feat_PCCon[i] = mean SciBERT embedding of TRAIN-ONLY contexts that cite paper i


Saves:
  feat_PCCon.pt    FloatTensor [N_total, 768]
"""

import json
from pathlib import Path

import torch

OUT_DIR = Path(".")

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading …")
with open(OUT_DIR / "node_index.json") as f:
    node_index = json.load(f)
N_total = len(node_index["paper"])
citation_id: dict[str, int] = node_index["citation"]

feat_C     = torch.load(OUT_DIR / "feat_C.pt")               # [N_cit, 768]
adj_CP_cited = torch.load(OUT_DIR / "adj_CP_cited.pt").coalesce()  # [N_cit, N_papers]

print(f"  feat_C        {tuple(feat_C.shape)}")
print(f"  adj_CP_cited  {tuple(adj_CP_cited.shape)}  nnz={adj_CP_cited._nnz():,}")

feat_C_nonzero = (feat_C.abs().sum(1) > 0).sum().item()
print(f"  feat_C non-zero rows: {feat_C_nonzero:,} / {feat_C.shape[0]:,} "
      f"(sanity check — should reflect TRAIN-only citations if step5a was "
      f"correctly filtered against split_uris.json)")

# ── Load frozen split (leakage guard) ─────────────────────────────────────────
print("Loading split_uris.json …")
with open(OUT_DIR / "split_uris.json") as f:
    split = json.load(f)
train_citing_uris: set[str] = set(split["train"])
print(f"  Train citing_uris: {len(train_citing_uris):,} "
      f"(val={len(split['val']):,}, test={len(split['test']):,})")

# ── Build the set of TRAIN citation-node integer IDs ──────────────────────────
# Same construction as step5a_encode_citations.py: citing_uri + citing_idx
# from all_contexts.json → citation node URI → citation_id lookup. Kept in
# lockstep with step5a on purpose; factor into a shared helper if step5a
# changes.
print("Loading all_contexts.json …")
with open("all_contexts.json") as f:
    all_contexts = json.load(f)

train_citation_int_ids: set[int] = set()
for entry in all_contexts:
    citing_uri = entry.get("citing_uri", "").strip()
    citing_idx = entry.get("citing_idx")
    if not citing_uri or citing_idx is None:
        continue
    if citing_uri not in train_citing_uris:
        continue
    cit_uri = citing_uri.replace("/paper/", "/citation/") + f"/{citing_idx}"
    if cit_uri in citation_id:
        train_citation_int_ids.add(citation_id[cit_uri])

print(f"  Train citation-node IDs: {len(train_citation_int_ids):,} / {len(citation_id):,}")

# ── Mask adj_CP_cited down to TRAIN citation-node rows only ──────────────────
# This is what fixes the leaking denominator: any nonzero entry whose row
# (citation-node index) is not in train_citation_int_ids is dropped BEFORE
# row_normalise, so the mean in feat_PCCon is computed over exactly the same
# contexts that feat_C actually has non-zero rows for.
print("Masking adj_CP_cited to train-only citation nodes …")
_indices = adj_CP_cited.indices()   # [2, nnz]
_values  = adj_CP_cited.values()    # [nnz]
_row_ids = _indices[0]

train_id_tensor = torch.tensor(sorted(train_citation_int_ids), dtype=torch.long)
_keep_mask = torch.isin(_row_ids, train_id_tensor)

adj_CP_cited_train = torch.sparse_coo_tensor(
    indices=_indices[:, _keep_mask],
    values=_values[_keep_mask],
    size=adj_CP_cited.shape,
).coalesce()

print(f"  adj_CP_cited        nnz={adj_CP_cited._nnz():,}")
print(f"  adj_CP_cited_train  nnz={adj_CP_cited_train._nnz():,}  "
      f"(dropped {adj_CP_cited._nnz() - adj_CP_cited_train._nnz():,} val/test edges)")

adj_CP_cited = adj_CP_cited_train

# ── row_normalise (copied from step4) ────────────────────────────────────────
def row_normalise(sp: torch.Tensor) -> torch.Tensor:
    sp = sp.coalesce()
    indices  = sp.indices()
    values   = sp.values()

    # Compute row degrees
    row_sum  = torch.zeros(sp.shape[0], dtype=torch.float)
    row_sum.scatter_add_(0, indices[0], values)

    # Avoid division by zero
    row_sum_safe = row_sum.clamp(min=1e-9)

    # Scale values
    new_values   = values / row_sum_safe[indices[0]]

    return torch.sparse_coo_tensor(indices, new_values, sp.shape).coalesce()

# ── Transpose adj_CP_cited → [N_papers, N_citations] ─────────────────────────
# Each row i now lists the citation nodes that cited paper i
adj_cited_T = adj_CP_cited.t().coalesce()   # [N_papers, N_citations]
print(f"  adj_CP_cited.T  {tuple(adj_cited_T.shape)}")

adj_cited_T_norm = row_normalise(adj_cited_T) # [N_papers, N_citations]
                                              # feat_PCCon[i] = mean of SciBERT embeddings of all contexts in which paper i was cited.
# ── Propagate ─────────────────────────────────────────────────────────────────
print("Propagating feat_PCCon …")
feat_PCCon = torch.sparse.mm(adj_cited_T_norm, feat_C)   # [N_papers, 768]

torch.save(feat_PCCon, OUT_DIR / "feat_PCCon.pt")
nonzero = (feat_PCCon.abs().sum(1) > 0).sum().item()
print(f"Saved feat_PCCon.pt  shape={tuple(feat_PCCon.shape)}")
print(f"  Non-zero rows: {nonzero} / {N_total}")