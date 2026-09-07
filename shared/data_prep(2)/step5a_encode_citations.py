"""
step5a_encode_citations.py
──────────────────────────
Reads:
  all_contexts.json    [{context, cited_uri, citing_uri, citing_idx}, ...]
  node_index.json      (citation_id map)
  split_uris.json      (frozen train/val/test citing_uri lists)

SciBERT-encodes each citation node's context passage (CLS token).
Citation nodes with no entry in all_contexts.json get a zero vector.

IMPORTANT (leakage fix): only citation contexts whose `citing_uri` is in the
FROZEN TRAIN split are encoded. Val/test citing papers' contexts are treated
as if they don't exist here, so downstream feat_PCCon can never be built
from — or contaminated by — val/test data. This mirrors real inference,
where "future" citation contexts wouldn't be available anyway.

Saves:
  feat_C.pt    FloatTensor [N_citations, 768]
"""

import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModel

OUT_DIR    = Path(".")
MODEL_NAME = "allenai/scibert_scivocab_uncased"
BATCH_SIZE = 32
MAX_LENGTH = 512
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# ── Load index ────────────────────────────────────────────────────────────────
print("Loading index …")
with open(OUT_DIR / "node_index.json") as f:
    node_index = json.load(f)
citation_id: dict[str, int] = node_index["citation"]
N_citations = len(citation_id)
print(f"  Citation nodes: {N_citations:,}")

# ── Load frozen split (leakage guard) ─────────────────────────────────────────
print("Loading split_uris.json …")
with open(OUT_DIR / "split_uris.json") as f:
    split = json.load(f)
train_citing_uris: set[str] = set(split["train"])
print(f"  Train citing_uris: {len(train_citing_uris):,} "
      f"(val={len(split['val']):,}, test={len(split['test']):,})")

# ── Load contexts ─────────────────────────────────────────────────────────────
print("Loading all_contexts.json …")
with open("all_contexts.json") as f:
    all_contexts = json.load(f)
print(f"  Total entries: {len(all_contexts):,}")

# ── Build (citation_uri → text) map ──────────────────────────────────────────
cit_texts: dict[str, str] = {}
skipped = 0
skipped_not_train = 0
for entry in all_contexts:
    citing_uri = entry.get("citing_uri", "").strip()
    citing_idx = entry.get("citing_idx")
    text       = entry.get("context",   "").strip()
    if not citing_uri or citing_idx is None or not text:
        skipped += 1
        continue
    # Leakage guard: only encode contexts belonging to the frozen TRAIN split.
    # Val/test citing_uris are skipped here so their context text can never
    # flow into feat_C.pt / feat_PCCon.pt.
    if citing_uri not in train_citing_uris:
        skipped_not_train += 1
        continue
    cit_uri = citing_uri.replace("/paper/", "/citation/") + f"/{citing_idx}"
    if cit_uri not in citation_id:
        skipped += 1
        continue
    cit_texts[cit_uri] = text

print(f"  Matched citation nodes      : {len(cit_texts):,}")
print(f"  Skipped (bad/missing entry) : {skipped:,}")
print(f"  Skipped (not in train split): {skipped_not_train:,}")

# ── Load SciBERT ──────────────────────────────────────────────────────────────
print(f"Loading SciBERT on {DEVICE} …")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()

def encode_texts(texts: list[str]) -> torch.Tensor:
    encoded = tokenizer(
        texts, 
        padding=True, 
        truncation=True,
        max_length=MAX_LENGTH, 
        return_tensors="pt",
    ).to(DEVICE)
    with torch.no_grad():
        out = model(**encoded)
    return out.last_hidden_state[:, 0, :].cpu()

# ── Encode in bulk ────────────────────────────────────────────────────────────
# Flatten into parallel (int_id, text) lists then batch-encode
cit_ids_list  = [citation_id[uri] for uri in cit_texts]
cit_text_list = [cit_texts[uri]   for uri in cit_texts]
N_matched     = len(cit_text_list)

print(f"Encoding {N_matched:,} passages …")
encoded_vecs = torch.zeros(N_matched, 768, dtype=torch.float)

for b_start in range(0, N_matched, BATCH_SIZE):
    b_end  = min(b_start + BATCH_SIZE, N_matched)
    vecs   = encode_texts(cit_text_list[b_start:b_end])
    encoded_vecs[b_start:b_end] = vecs
    if (b_start // BATCH_SIZE) % 20 == 0:
        print(f"  {b_end}/{N_matched}", end="\r")

print()

# ── Scatter into full [N_citations, 768] tensor ───────────────────────────────
feat_C = torch.zeros(N_citations, 768, dtype=torch.float)
targets = torch.tensor(cit_ids_list, dtype=torch.long)
feat_C[targets] = encoded_vecs

torch.save(feat_C, OUT_DIR / "feat_C.pt")
print(f"Saved feat_C.pt  shape={tuple(feat_C.shape)}")
print(f"  Non-zero rows: {(feat_C.abs().sum(1) > 0).sum().item()} / {N_citations}")