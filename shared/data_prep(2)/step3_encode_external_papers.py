"""
step3_encode_external_papers.py
────────────────────────────────
Reads:
  merged-kg.ttl      (dcterms:title per paper)
  node_index.json
  external_ids.pt
  paper_uris.json

Title is guaranteed to exist for every paper in the knowledge graph, so
each external paper's feature is simply SciBERT(CLS) of its dcterms:title.
No abstract or citation-context text is used for external papers.

Saves:
  feat_external_papers.pt   FloatTensor [N_external, 768]
"""

import json
from pathlib import Path

import torch
from rdflib import Graph, Namespace
from transformers import AutoTokenizer, AutoModel

DCTERMS = Namespace("http://purl.org/dc/terms/")

KG_FILE    = "merged-kg.ttl"
OUT_DIR    = Path(".")
MODEL_NAME = "allenai/scibert_scivocab_uncased"
BATCH_SIZE = 32
MAX_LENGTH = 512
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

SPARQL_PREFIXES = """
PREFIX dcterms: <http://purl.org/dc/terms/>
"""

# ── Load index ────────────────────────────────────────────────────────────────
print("Loading index …")
with open(OUT_DIR / "node_index.json") as f:
    node_index = json.load(f)
paper_id: dict[str, int] = node_index["paper"]

with open(OUT_DIR / "paper_uris.json") as f:
    paper_uri_list: list[str] = json.load(f)   # index → URI

external_ids: torch.Tensor = torch.load(OUT_DIR / "external_ids.pt")
external_int_ids = external_ids.tolist()
external_uris    = [paper_uri_list[i] for i in external_int_ids]
N_external       = len(external_uris)
print(f"  External papers: {N_external}")

# ── Load titles (dcterms:title) from the KG via SPARQL ────────────────────────
print(f"Loading titles from {KG_FILE} …")
g = Graph()
g.parse(KG_FILE, format="turtle")

q_title = SPARQL_PREFIXES + """
SELECT ?s ?o WHERE { ?s dcterms:title ?o . }
"""

titles: dict[str, str] = {}
for s, o in g.query(q_title):
    uri = str(s)
    text = str(o).strip()
    if text:
        titles[uri] = text
print(f"  Title entries: {len(titles):,}")

n_missing = sum(1 for uri in external_uris if uri not in titles)
if n_missing:
    print(f"  [WARN] {n_missing} external papers have no dcterms:title "
          f"(expected none — titles are guaranteed in the KG).")

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

# ── Encode titles in batches ───────────────────────────────────────────────────
print("Encoding …")
feat = torch.zeros(N_external, 768, dtype=torch.float)

for batch_start in range(0, N_external, BATCH_SIZE):
    batch_uris  = external_uris[batch_start : batch_start + BATCH_SIZE]
    batch_texts = [titles.get(uri, "").strip() for uri in batch_uris]

    vecs = encode_texts(batch_texts)   # [len(batch_uris), 768]
    feat[batch_start : batch_start + len(batch_uris)] = vecs

    done = min(batch_start + BATCH_SIZE, N_external)
    print(f"  {done}/{N_external}", end="\r")

print()

# ── Save ──────────────────────────────────────────────────────────────────────
torch.save(feat, OUT_DIR / "feat_external_papers.pt")
print(f"Saved feat_external_papers.pt  shape={tuple(feat.shape)}")