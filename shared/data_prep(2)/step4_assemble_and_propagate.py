"""
step4_assemble_and_propagate.py
────────────────────────────────
Reads:
  feat_corpus_papers.pt     [N_corpus, 768]
  feat_external_papers.pt   [N_external, 768]
  node_index.json
  corpus_ids.pt
  external_ids.pt
  adj_PP.pt                 [N_papers, N_papers]  raw (unweighted)
  adj_CP_citing.pt          [N_citations, N_papers]

Produces:
  feat_P.pt     [N_total, 768]  raw stacked features
  feat_PP.pt    [N_total, 768]  1-hop citation-neighbour mean
"""

import json
from pathlib import Path

import torch

# if needed for comptability with torch_sparse.SparseTensor
# def to_torch_sparse_tensor(sp: torch.Tensor):
#     """Convert native PyTorch sparse COO to torch_sparse.SparseTensor if needed."""
#     try:
#         import torch_sparse
#         sp = sp.coalesce().cpu()
#         row, col = sp.indices()
#         value = sp.values()
#         return torch_sparse.SparseTensor(
#             row=row, col=col, value=value, 
#             sparse_sizes=sp.shape
#         )
#     except ImportError:
#         print("torch_sparse not installed — keeping native sparse tensors")
#         return sp

        
OUT_DIR = Path(".")

# ── Load index ────────────────────────────────────────────────────────────────
print("Loading index …")
with open(OUT_DIR / "node_index.json") as f:
    node_index = json.load(f)
paper_id: dict[str, int] = node_index["paper"]
N_total = len(paper_id)

corpus_ids   = torch.load(OUT_DIR / "corpus_ids.pt")    # [N_corpus]  LongTensor
external_ids = torch.load(OUT_DIR / "external_ids.pt")  # [N_external] LongTensor
N_corpus   = len(corpus_ids)
N_external = len(external_ids)

# ── Load partial feature tensors ─────────────────────────────────────────────
print("Loading partial features …")
feat_corpus   = torch.load(OUT_DIR / "feat_corpus_papers.pt")    # [N_corpus, 768]
feat_external = torch.load(OUT_DIR / "feat_external_papers.pt")  # [N_external, 768]

assert feat_corpus.shape   == (N_corpus,   768), f"Unexpected shape: {feat_corpus.shape}"
assert feat_external.shape == (N_external, 768), f"Unexpected shape: {feat_external.shape}"

# ── Assemble full feature matrix ──────────────────────────────────────────────
# corpus_ids and external_ids carry the global int IDs produced in step 1.
# Scatter each block into the right rows of feat_P.
print("Assembling feat_P …")
feat_P = torch.zeros(N_total, 768, dtype=torch.float)
feat_P[corpus_ids]   = feat_corpus
feat_P[external_ids] = feat_external
torch.save(feat_P, OUT_DIR / "feat_P.pt")
print(f"  feat_P  shape={tuple(feat_P.shape)}")

# ── Load adjacency matrices ───────────────────────────────────────────────────
print("Loading adjacency matrices …")
adj_PP        = torch.load(OUT_DIR / "adj_PP.pt").coalesce()         # [N, N]
adj_CP_citing = torch.load(OUT_DIR / "adj_CP_citing.pt").coalesce()  # [C, N]

# Convert for compatibility (not sure of it) 
# adj_PP        = to_torch_sparse_tensor(adj_PP)
# adj_CP_citing = to_torch_sparse_tensor(adj_CP_citing)

N_citations = adj_CP_citing.shape[0]
print(f"  adj_PP        {tuple(adj_PP.shape)}  nnz={adj_PP._nnz():,}")
print(f"  adj_CP_citing {tuple(adj_CP_citing.shape)}  nnz={adj_CP_citing._nnz():,}")

# ── Helper: row-normalise a sparse COO tensor ─────────────────────────────────
def row_normalise(sp: torch.Tensor) -> torch.Tensor:
    """
    Divide each non-zero by the row sum.
    Rows with degree 0 stay all-zero (no-op, avoids division by zero).
    Returns a new sparse COO tensor.
    """
    sp = sp.coalesce()
    indices = sp.indices()   # [2, nnz]
    values  = sp.values()    # [nnz]

    # Compute row degrees
    row_sum = torch.zeros(sp.shape[0], dtype=torch.float)
    row_sum.scatter_add_(0, indices[0], values)

    # Avoid division by zero
    row_sum_safe = row_sum.clamp(min=1e-9)

    # Scale values
    new_values = values / row_sum_safe[indices[0]]

    return torch.sparse_coo_tensor(indices, new_values, sp.shape).coalesce()

# ── Row-normalise adjacency ────────────────────────────────────────────────────
print("Row-normalising …")
adj_PP_norm = row_normalise(adj_PP)

# ── 1-hop propagation: feat_PP = adj_PP_norm @ feat_P ────────────────────────
#
# Each paper's new feature = mean of its cited neighbours' raw features.
# Papers with no outgoing citations (external papers, Category C) → zero row
# in adj_PP_norm → feat_PP row stays zero (handled by row_normalise guard).
#
print("Propagating feat_PP …")
# torch.sparse mm: sparse [N,N] × dense [N,768] → dense [N,768]
feat_PP = torch.sparse.mm(adj_PP_norm, feat_P)
torch.save(feat_PP, OUT_DIR / "feat_PP.pt")
print(f"  feat_PP  shape={tuple(feat_PP.shape)}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\nAll done. Output files:")
for name in ["feat_P.pt", "feat_PP.pt"]:
    p = OUT_DIR / name
    t = torch.load(p)
    nonzero_rows = (t.abs().sum(dim=1) > 0).sum().item()
    print(f"  {name:20s}  shape={tuple(t.shape)}  "
          f"non-zero rows={nonzero_rows}/{t.shape[0]}")