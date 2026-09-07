"""
step2_encode_corpus_papers_sparql.py
─────────────────────────────────────


One SPARQL query is run against the graph:
  - ?s dcterms:title ?o → gives each paper's title text

The result rows are used to build the `titles` lookup dict (URI → title
text), which is then merged with each paper's abstract before SciBERT
encoding. Everything else (abstract loading, encoding, saving) is
unchanged. Output is identical in content and format.

Reads:
  abstracts.json   {paper_uri: abstract_text}
  merged-kg.ttl    (dcterms:title per paper, used for the title text)
  node_index.json  (paper_id map)
  corpus_ids.pt    (which int IDs are corpus papers)
  paper_uris.json  (int_id → URI lookup)

SciBERT-encodes "{abstract} [SEP] {title}" (CLS token) in batches.
Title is appended after the abstract; if a paper has no dcterms:title the
abstract is encoded alone. Papers with no abstract entry get a zero vector
and are flagged (title alone is not sufficient to produce a feature).

Saves:
  feat_corpus_papers.pt        FloatTensor [N_corpus, 768]
  corpus_missing_abstract.json list of URIs with no abstract
"""

import json
from pathlib import Path

import torch
from rdflib import Graph, Namespace
from transformers import AutoTokenizer, AutoModel

DCTERMS = Namespace("http://purl.org/dc/terms/")

ABSTRACTS_FILE = "abstracts.json"
KG_FILE        = "merged-kg.ttl"
OUT_DIR        = Path(".")
MODEL_NAME     = "allenai/scibert_scivocab_uncased"
BATCH_SIZE     = 32
MAX_LENGTH     = 512 #SciBERT's max input length is 512 tokens cap
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

SPARQL_PREFIXES = """
PREFIX dcterms: <http://purl.org/dc/terms/>
"""

# ── Load index structures ─────────────────────────────────────────────────────
print("Loading index …")
with open(OUT_DIR / "node_index.json") as f:
    node_index = json.load(f)
paper_id: dict[str, int] = node_index["paper"]

with open(OUT_DIR / "paper_uris.json") as f:
    paper_uri_list: list[str] = json.load(f)   # index → URI

corpus_ids: torch.Tensor = torch.load(OUT_DIR / "corpus_ids.pt")
corpus_int_ids = corpus_ids.tolist()
corpus_uris    = [paper_uri_list[i] for i in corpus_int_ids]
N_corpus       = len(corpus_uris)
print(f"  Corpus papers : {N_corpus}")

# ── Load abstracts ────────────────────────────────────────────────────────────
print("Loading abstracts …")
with open(ABSTRACTS_FILE) as f:
    abstracts: dict[str, str] = json.load(f)
print(f"  Abstract entries: {len(abstracts):,}")

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

n_corpus_missing_title = sum(1 for uri in corpus_uris if uri not in titles)
if n_corpus_missing_title:
    print(f"  [WARN] {n_corpus_missing_title} corpus papers have no dcterms:title — "
          f"abstract will be used alone for those.")

# ── Load SciBERT ──────────────────────────────────────────────────────────────
print(f"Loading SciBERT on {DEVICE} …")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()

# ── Encode in batches ─────────────────────────────────────────────────────────
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
    # CLS token is position 0
    return out.last_hidden_state[:, 0, :].cpu()

feat = torch.zeros(N_corpus, 768, dtype=torch.float)
missing: list[str] = []

print("Encoding …")
for batch_start in range(0, N_corpus, BATCH_SIZE):
    batch_uris   = corpus_uris[batch_start : batch_start + BATCH_SIZE]
    batch_texts  = []
    batch_local_idx = []   # positions within this batch that have real text

    for local_i, uri in enumerate(batch_uris):
        abstract_text = abstracts.get(uri, "").strip()
        if not abstract_text:
            missing.append(uri)
            continue

        title_text = titles.get(uri, "").strip()
        # Title appended at the end of the abstract, separated by SciBERT's
        # [SEP] token so the encoder still sees it as a distinct segment
        # rather than a run-on sentence.
        text = f"{abstract_text} [SEP] {title_text}" if title_text else abstract_text

        batch_texts.append(text)
        batch_local_idx.append(local_i)

    if batch_texts:
        vecs = encode_texts(batch_texts)   # [len(batch_texts), 768]
        for vec_i, local_i in enumerate(batch_local_idx):
            global_i = batch_start + local_i
            feat[global_i] = vecs[vec_i]

    done = min(batch_start + BATCH_SIZE, N_corpus)
    print(f"  {done}/{N_corpus}", end="\r")

print()
print(f"  Missing abstracts: {len(missing)}")

# ── Save ──────────────────────────────────────────────────────────────────────
torch.save(feat, OUT_DIR / "feat_corpus_papers.pt")
with open(OUT_DIR / "corpus_missing_abstract.json", "w") as f:
    f.write('[\n')
    for i, uri in enumerate(missing):
        comma = "," if i < len(missing) - 1 else ""
        f.write(f'  {json.dumps(uri)}{comma}\n')
    f.write(']\n')

print(f"Saved feat_corpus_papers.pt  shape={tuple(feat.shape)}")