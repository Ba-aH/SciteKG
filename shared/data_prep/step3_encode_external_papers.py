"""
step3_encode_external_papers.py
────────────────────────────────
Reads:
  merged.ttl         (for citekg:hasCitationContext → c4o:hasContext, and dcterms:title for the fallback)
  node_index.json
  external_ids.pt
  paper_uris.json
  split_uris.json    (frozen train/val/test citing_uri lists — LEAKAGE GUARD)

For each external paper:
  - Collect c4o:hasContext passage texts from its citekg:hasCitationContext
    citation nodes, but ONLY for citation nodes whose citing paper
    (cito:hasCitingEntity) is in the FROZEN TRAIN split. Val/test citing
    papers' contexts are treated as if they don't exist here, so
    feat_external_papers.pt (which becomes feat_P for external papers) can
    never be built from — or contaminated by — val/test citation contexts.
    This mirrors the guard already applied in step5a_encode_citations.py.
  - If ≥1 train context: feature = mean SciBERT(CLS) over those passages
  - If 0 train contexts (Category C): feature = SciBERT(CLS) of the paper's dcterms:title.
    A title is known for every paper and carries no val/test information, so this is not a
    leakage risk. Only a paper with neither a train context nor a title stays a zero vector.

Saves:
  feat_external_papers.pt      FloatTensor [N_external, 768]
  external_has_feature.pt      BoolTensor  [N_external]  (True = built from ≥1 train context; False = title fallback, or zero if no title)
  external_context_counts.json {uri: n_contexts}  — useful for diagnostics
  external_title_fallback.json URIs of Category C papers encoded from their title alone
"""

import json
from pathlib import Path

import torch
from rdflib import Graph, Namespace
from transformers import AutoTokenizer, AutoModel

CITEKG = Namespace("https://citekg.org/ontology/")
C4O    = Namespace("http://purl.org/spar/c4o/")
DCTERMS = Namespace("http://purl.org/dc/terms/")

OUT_DIR    = Path(".")
MODEL_NAME = "allenai/scibert_scivocab_uncased"
BATCH_SIZE = 32
MAX_LENGTH = 512
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# ── Load graph (needed for context passage text) ──────────────────────────────
print("Loading merged.ttl …")
g = Graph()
g.parse("merged-kg.ttl", format="turtle")
print(f"  {len(g):,} triples")

# ── Load index ────────────────────────────────────────────────────────────────
print("Loading index …")
with open(OUT_DIR / "node_index.json") as f:
    node_index = json.load(f)
paper_id: dict[str, int] = node_index["paper"]

with open(OUT_DIR / "paper_uris.json") as f:
    paper_uri_list: list[str] = json.load(f)

external_ids: torch.Tensor = torch.load(OUT_DIR / "external_ids.pt")
external_int_ids = external_ids.tolist()
external_uris    = [paper_uri_list[i] for i in external_int_ids]
N_external       = len(external_uris)
print(f"  External papers: {N_external}")

# ── Load frozen split (leakage guard) ─────────────────────────────────────────
# Same guard as step5a_encode_citations.py: only citation contexts whose
# citing paper is in the FROZEN TRAIN split are allowed to contribute to
# feat_external_papers.pt. Without this, an external paper's identity
# feature would be built from val/test citation context text, which is
# exactly what the model is later asked to retrieve against.
print("Loading split_uris.json …")
with open(OUT_DIR / "split_uris.json") as f:
    split = json.load(f)
train_citing_uris: set[str] = set(split["train"])
print(f"  Train citing_uris: {len(train_citing_uris):,} "
      f"(val={len(split['val']):,}, test={len(split['test']):,})")

# ── Build citation node → citing paper map (for the train-only filter) ───────
# Mirrors adj_CP_citing construction in step1_build_graph.py
# (cito:hasCitingEntity : citation_node -> citing_paper)
print("Collecting citation-node → citing-paper map …")
from rdflib import URIRef

CITO = Namespace("http://purl.org/spar/cito/")
cit_node_to_citing_uri: dict = {}
for cit_node, _, citing_paper in g.triples((None, CITO.hasCitingEntity, None)):
    cit_node_to_citing_uri[cit_node] = str(citing_paper)

# ── Build URI → context passages map from the graph (TRAIN-ONLY) ─────────────
# citekg:hasCitationContext on the cited paper points to citation nodes;
# each citation node has c4o:hasContext with the passage text. A passage is
# only kept if its citation node's citing paper is in train_citing_uris.
print("Collecting context passages for external papers (train-only) …")

ext_contexts: dict[str, list[str]] = {uri: [] for uri in external_uris}
ext_uri_set = set(external_uris)

n_skipped_not_train = 0
n_skipped_no_citing  = 0

for paper_uri_str in external_uris:
    paper_ref = URIRef(paper_uri_str)
    for _, _, cit_node in g.triples((paper_ref, CITEKG.hasCitationContext, None)):
        citing_uri = cit_node_to_citing_uri.get(cit_node)
        if citing_uri is None:
            n_skipped_no_citing += 1
            continue
        if citing_uri not in train_citing_uris:
            n_skipped_not_train += 1
            continue
        # Get the passage text from this citation node
        for _, _, ctx_lit in g.triples((cit_node, C4O.hasContext, None)):
            text = str(ctx_lit).strip()
            if text:
                ext_contexts[paper_uri_str].append(text)

context_counts = {uri: len(v) for uri, v in ext_contexts.items()}
n_with_context = sum(1 for v in context_counts.values() if v > 0)
print(f"  External with ≥1 train context : {n_with_context}")
print(f"  External with 0 train contexts (Category C) : {N_external - n_with_context}")
print(f"  Contexts skipped (not in train split) : {n_skipped_not_train:,}")
print(f"  Contexts skipped (no citing entity found) : {n_skipped_no_citing:,}")

# ── Load SciBERT ──────────────────────────────────────────────────────────────
print(f"Loading SciBERT on {DEVICE} …")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()

def encode_texts(texts: list[str]) -> torch.Tensor:
    """Returns [len(texts), 768] CLS embeddings."""
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

# ── Encode: for each external paper, mean-pool its context passages ───────────
print("Encoding …")
feat        = torch.zeros(N_external, 768, dtype=torch.float)
has_feature = torch.zeros(N_external, dtype=torch.bool)

for ext_idx, uri in enumerate(external_uris):
    passages = ext_contexts[uri]
    if not passages:
        # Category C — handled by the title fallback below, flag stays False
        continue

    # Encode in sub-batches (a paper can have many contexts)
    all_vecs = []
    for b_start in range(0, len(passages), BATCH_SIZE):
        batch = passages[b_start : b_start + BATCH_SIZE]
        all_vecs.append(encode_texts(batch))

    paper_vecs  = torch.cat(all_vecs, dim=0)  # [n_ctx, 768]
    feat[ext_idx]        = paper_vecs.mean(dim=0) # mean pooling here the N passages of paper[i]
    has_feature[ext_idx] = True

    if (ext_idx + 1) % 100 == 0:
        print(f"  {ext_idx + 1}/{N_external}", end="\r")

print(f"\n  Encoded {has_feature.sum().item()} / {N_external} external papers")

# ── Fallback for Category C: encode the paper's title ─────────────────────────
# A paper with 0 train contexts has no content from the context side. Its own title is always
# available (it is not a val/test citation context), so it is encoded the same way as everything
# else (SciBERT CLS) and written into the same feat_P row. has_feature stays False for these papers.
title_only: list[str] = []      # Category C papers encoded from their title
no_title:   list[str] = []      # Category C papers with no title either (stay zero)
fallback_idx, fallback_texts = [], []
for ext_idx, uri in enumerate(external_uris):
    if has_feature[ext_idx]:
        continue
    title = g.value(URIRef(uri), DCTERMS.title)
    title = str(title).strip() if title is not None else ""
    if title:
        fallback_idx.append(ext_idx)
        fallback_texts.append(title)
        title_only.append(uri)
    else:
        no_title.append(uri)

for b_start in range(0, len(fallback_texts), BATCH_SIZE):
    feat[fallback_idx[b_start : b_start + BATCH_SIZE]] = encode_texts(fallback_texts[b_start : b_start + BATCH_SIZE])
print(f"  Title fallback (0 train contexts, title used): {len(title_only)}")
print(f"  Still zero (0 train contexts AND no title)   : {len(no_title)}")

# ── Save ──────────────────────────────────────────────────────────────────────
torch.save(feat,        OUT_DIR / "feat_external_papers.pt")
torch.save(has_feature, OUT_DIR / "external_has_feature.pt")
with open(OUT_DIR / "external_context_counts.json", "w") as f:
    items = list(context_counts.items())
    f.write('{\n')
    for i, (uri, count) in enumerate(items):
        comma = "," if i < len(items) - 1 else ""
        f.write(f'  {json.dumps(uri)}: {count}{comma}\n')
    f.write('}\n')

with open(OUT_DIR / "external_title_fallback.json", "w") as f:
    f.write('[\n')
    for i, uri in enumerate(title_only):
        comma = "," if i < len(title_only) - 1 else ""
        f.write(f'  {json.dumps(uri)}{comma}\n')
    f.write(']\n')

print(f"Saved feat_external_papers.pt  shape={tuple(feat.shape)}")
print(f"Saved external_has_feature.pt  True={has_feature.sum().item()}")