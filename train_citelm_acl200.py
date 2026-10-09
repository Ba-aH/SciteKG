"""
train_citelm_acl200.py
----------------------
Autoregressive citation-vector LM  -  ACL-200 (baseline dataset) version.

ADAPTED from the ArgKG version (train_cite_lm.py). What changed and why:
  - Everything is imported from train_baseline (NOT train_domain_specific), so the
    ACL-200 `dataset.py` (shared/acl200_prep) is used instead of the ArgKG one
    (shared/data_prep). Importing the ArgKG module was the cause of
    "0 records kept, 63,316 skipped".
  - GroupAwareBatchSampler removed: ACL-200 has no context_group_id / multi-citation
    grouping (every citation context cites exactly one paper), so a plain shuffled
    DataLoader is used.
  - Candidate pool = ALL papers (arange over node_index.json["paper"]), exactly like
    train_baseline.py, so the metrics are directly comparable with the two-tower baseline.
  - evaluate() is the per-record version from train_baseline (one query per record).

Autoregressive citation-vector LM (unchanged below this line).

The model reads a citation context left-to-right (causal LM, teacher forcing),
then at a final [CITE_VEC] position it emits a vector in the SAME space as the
frozen PaperTower embeddings. Retrieval = nearest neighbour over those embeddings.

Sequence layout (citation marker moved to the END of the context):
    [CLS] w1 w2 ... wn [CITE_VEC]

Losses:
    LM loss        : next-token cross-entropy on w1..wn and on [CITE_VEC]
                     (so [CITE_VEC] doubles as an "end of passage -> cite now" signal)
    Alignment loss : InfoNCE of the vector read at [CITE_VEC] against the FULL frozen
                     candidate pool (paper embeddings never change, so a full softmax
                     is cheap and matches evaluation exactly)
    total = alignment + lambda_lm * LM        (lambda_lm = 0  ->  alignment-only ablation)

Reuses from your ACL-200 two-tower script (so metrics are identical to the baseline):
    evaluate, build_candidate_index, PaperTower, build_datasets, lcr_collate_fn, load_config

Usage:
    python train_citelm_acl200.py --config configs/cite_lm_acl200/align_only.json
    python train_citelm_acl200.py --config configs/cite_lm_acl200/align_lm.json

    nohup python train_citelm_acl200.py --config configs/cite_lm_acl200/align_lm.json >myoutfile 2>&1 &
"""

# ===========================================================================
# HOW TO READ THIS FILE
# ---------------------------------------------------------------------------
# The file has five parts, in this order:
#   1. Imports
#   2. CitationLM            - the model (preparing inputs, encoding, LM loss)
#   3. load_frozen_paper_index - builds the fixed paper-embedding "answer sheet"
#   4. train_one_epoch       - one pass over the training data
#   5. main                  - wires everything together (setup -> train -> test)
#
# Shape notation used in the comments:
#   B = batch size, L = sequence length (tokens), H = hidden size (768 for SciBERT),
#   d = embed_dim (size of the paper embeddings), N = number of candidate papers,
#   V = vocabulary size.
# ===========================================================================

# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------
import os                          # file paths (os.path.join, expanduser) and folder creation
import json                        # saving config / history files
import math                        # math.isnan / math.isinf for the NaN guard
import time                        # timing epochs and the whole run
import random                      # python's RNG (seeded for reproducibility)
import argparse                    # command-line arguments (--config)

import numpy as np                 # numpy's RNG (seeded for reproducibility)
import torch                       # tensors, autograd, GPU
import torch.nn as nn              # layers (Linear, LayerNorm, Dropout) and nn.Module
import torch.nn.functional as F    # functional ops (cross_entropy, normalize, softmax)
from torch.utils.data import DataLoader   # batches a dataset
from tqdm import tqdm              # progress bars
from transformers import AutoTokenizer, BertLMHeadModel
# AutoTokenizer  : loads the SciBERT tokenizer (text <-> token ids)
# BertLMHeadModel: BERT + a language-modelling head; with is_decoder=True it uses a CAUSAL mask

# Importing this module also does the sys.path setup (shared/acl200_prep -> ACL-200 dataset.py).
# If you rename train_baseline.py (e.g. to train_twotower_acl200.py), change the module name here.
# IMPORTANT: do NOT import train_domain_specific in this file - it puts the ArgKG dataset.py first on sys.path.
from train_baseline import (
    load_config,              # reads the JSON config file into a dict
    evaluate,                 # ranks all candidates for every val/test context -> Recall@K, MRR, nDCG
    build_candidate_index,    # runs PaperTower over all candidate papers -> [N, d] embedding matrix
    PaperTower,               # the paper encoder (SeHGNN-style) you already trained
    build_datasets,           # builds train/val/test datasets of (context, cited paper) records
    lcr_collate_fn,           # turns a list of records into one batch of tensors
)


# ---------------------------------------------------------------------------
# 2. Model
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# CitationLM (class)
#
# Wraps SciBERT used as a CAUSAL language model (each token only sees tokens to
# its left) and adds a small head that turns the hidden state at the final
# [CITE_VEC] token into a vector in the paper-embedding space.
#
# Two ways to use it:
#   - model(input_ids, attention_mask)  -> citation vectors [B, d]. Same signature as your
#                                          ContextTower, so evaluate() works on it unchanged.
#   - model.forward_train(...)          -> also returns the LM loss, for training.
# ---------------------------------------------------------------------------
class CitationLM(nn.Module):
    """
    SciBERT initialised as a CAUSAL LM (is_decoder=True) + a citation-vector head.

    forward(input_ids, attention_mask) has the same signature as ContextTower, so
    your existing `evaluate()` works on it unchanged. It takes the tokenised
    contexts produced by your dataset, removes the in-text citation marker,
    appends [CITE_VEC] at the end and returns the L2-normalised vector read there.
    """

    # -----------------------------------------------------------------------
    # __init__
    #
    # Builds all the parts of the model once:
    #   - the tokenizer (with one extra special token [CITE_VEC]),
    #   - the token-id patterns of the in-text markers we must strip,
    #   - the causal SciBERT language model,
    #   - the citation head (Dropout -> Linear -> LayerNorm).
    #
    # Arguments:
    #   model_name     : HuggingFace name of SciBERT ("allenai/scibert_scivocab_uncased")
    #   embed_dim      : size d of the paper embeddings the vector must live in
    #   dropout        : dropout probability applied before the citation head
    #   marker_strings : list of marker texts to delete from the context, e.g. ["[cite]"]
    # -----------------------------------------------------------------------
    def __init__(self, model_name: str, embed_dim: int, dropout: float, marker_strings: list):
        super().__init__()   # initialise nn.Module internals (needed before registering layers)

        # Load the SciBERT tokenizer (turns text into token ids and back).
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Token-id sequences of the in-text markers to strip (e.g. "[cite]" -> "[", "cite", "]").
        # Computed BEFORE adding the new special token so nothing collides.
        # For each marker string: tokenize it WITHOUT [CLS]/[SEP] and keep the list of ids.
        # Result: a list of id-lists, e.g. [[1646, 4326, 1648]].
        self.marker_seqs = [
            self.tokenizer(m, add_special_tokens=False)["input_ids"] for m in marker_strings
        ]

        # Register a brand-new special token "[CITE_VEC]". It gets the next free id in the vocabulary.
        # This token marks the slot where the model outputs the citation vector.
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["[CITE_VEC]"]})
        # Look up the id that was just assigned to [CITE_VEC].
        self.cite_id = self.tokenizer.convert_tokens_to_ids("[CITE_VEC]")
        # Id of the padding token (used to right-pad shorter sequences in a batch).
        self.pad_id = self.tokenizer.pad_token_id
        # Id of [CLS] (always the first token; here it also acts as the "start of text" token).
        self.cls_id = self.tokenizer.cls_token_id

        # Causal attention mask comes from is_decoder=True
        # Load SciBERT weights into a BERT-with-LM-head model. is_decoder=True makes the
        # attention causal: position t can only attend to positions <= t. Without this the
        # model could see the future tokens and the LM loss would be trivial (cheating).
        self.lm = BertLMHeadModel.from_pretrained(model_name, is_decoder=True)
        # The vocabulary grew by one token ([CITE_VEC]), so enlarge the embedding matrix
        # (and the tied output layer) by one row. The new row starts randomly initialised.
        self.lm.resize_token_embeddings(len(self.tokenizer))

        # Hidden size H of the model (768 for SciBERT). Read from the config, not hard-coded.
        hidden = self.lm.config.hidden_size
        # Randomly zeroes some hidden dimensions during training to reduce overfitting.
        self.dropout = nn.Dropout(dropout)
        # Linear projection H -> d: maps the hidden state into the paper-embedding space.
        self.proj = nn.Linear(hidden, embed_dim)
        # LayerNorm over the d dimensions: stabilises activations before L2 normalisation
        # (same recipe as your ContextTower).
        self.norm = nn.LayerNorm(embed_dim)
        # Xavier-uniform init of the projection weights (gain matched to ReLU, as in ContextTower).
        nn.init.xavier_uniform_(self.proj.weight, gain=nn.init.calculate_gain("relu"))
        # Start the projection bias at zero.
        nn.init.zeros_(self.proj.bias)

    # -----------------------------------------------------------------------
    # _strip_markers
    #
    # Removes every occurrence of the marker token-patterns (e.g. "[", "cite", "]")
    # from a list of token ids and returns the cleaned list.
    # It scans left to right: at each position it checks whether any marker pattern
    # starts there; if so it skips over it, otherwise it keeps the token.
    # Why: the real citation marker used to sit in the middle of the sentence. We
    # move the "citation slot" to the end, so the original marker must be deleted
    # (otherwise the model would see where / how the paper was cited).
    # -----------------------------------------------------------------------
    def _strip_markers(self, ids: list) -> list:
        out, i = [], 0                                    # out = cleaned ids, i = scan position
        while i < len(ids):                               # walk over every token
            for seq in self.marker_seqs:                  # try each marker pattern at position i
                if ids[i : i + len(seq)] == seq:          # does this pattern start exactly here?
                    i += len(seq)                         # yes: jump over the whole marker (drop it)
                    break                                 # stop trying other patterns at this spot
            else:                                         # runs only if the for loop did NOT break,
                out.append(ids[i])                        # i.e. no marker here: keep this token
                i += 1                                    # and move to the next token
        return out                                        # context ids without any marker

    # -----------------------------------------------------------------------
    # prepare
    #
    # Converts the batch produced by your dataset into the layout this model needs:
    #     [CLS] body [SEP] (+padding)   ->   [CLS] body_without_marker [CITE_VEC] (+padding)
    # i.e. it (1) cuts off the padding, (2) drops [CLS]/[SEP], (3) strips the markers,
    # (4) puts [CLS] back at the front and [CITE_VEC] at the end, (5) re-pads the batch.
    # It also returns cite_pos: the index of [CITE_VEC] in each row, which is where we
    # read the citation vector.
    # Runs under no_grad because it only manipulates integer ids (nothing to differentiate).
    # -----------------------------------------------------------------------
    @torch.no_grad()
    def prepare(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        [CLS] body [SEP] (+pad)  ->  [CLS] body_without_marker [CITE_VEC] (+pad)
        Returns (ids, mask, cite_pos) on the same device as the input.
        Assumes your dataset tokenisation is [CLS] ... [SEP] (standard BERT tokenizer output).
        """
        device = input_ids.device                         # remember GPU/CPU to send results back there
        rows = []                                         # one cleaned token list per example
        for ids, m in zip(input_ids.cpu(), attention_mask.cpu()):   # loop over the batch on CPU (python lists are easier)
            toks = ids[m.bool()].tolist()                 # keep only real tokens (mask==1), drop padding
            body = self._strip_markers(toks[1:-1])        # drop [CLS] and [SEP]
            rows.append([self.cls_id] + body + [self.cite_id])   # new row = [CLS] body [CITE_VEC]

        L = max(len(r) for r in rows)                     # longest new row = padded length of the batch
        new_ids = torch.full((len(rows), L), self.pad_id, dtype=torch.long)    # [B, L] filled with PAD
        new_mask = torch.zeros((len(rows), L), dtype=torch.long)               # [B, L] filled with 0 (= padding)
        for b, r in enumerate(rows):                      # copy each row into the padded tensors
            new_ids[b, : len(r)] = torch.tensor(r)        # real tokens go at the left, PAD stays on the right
            new_mask[b, : len(r)] = 1                     # mark the real tokens with 1
        cite_pos = new_mask.sum(dim=1) - 1                # index of [CITE_VEC] per row
        # (number of real tokens - 1 = index of the LAST real token = [CITE_VEC])
        return new_ids.to(device), new_mask.to(device), cite_pos.to(device)   # back to the original device

    # -----------------------------------------------------------------------
    # _encode
    #
    # Runs the causal SciBERT on the prepared batch, picks the hidden state at the
    # [CITE_VEC] position of every row, and turns it into the L2-normalised
    # citation vector. Also returns ALL hidden states (needed for the LM loss).
    # Returns: (vec [B, d], h [B, L, H])
    # -----------------------------------------------------------------------
    def _encode(self, ids, mask, cite_pos):
        # self.lm.bert is the BERT body WITHOUT the LM head. Output: one H-dim vector per token.
        # use_cache=False: no key/value caching (we are training, caching would waste memory).
        h = self.lm.bert(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state   # [B, L, H]
        # For row b, take h[b, cite_pos[b]]: the hidden state at the [CITE_VEC] token.
        # Thanks to the causal mask this vector has seen the ENTIRE context (and nothing after it).
        h_cite = h[torch.arange(h.size(0), device=h.device), cite_pos]       # [B, hidden]
        x = self.norm(self.proj(self.dropout(h_cite)))    # dropout -> Linear(H -> d) -> LayerNorm
        return F.normalize(x, p=2, dim=-1), h             # L2-normalise: unit length, dot product = cosine

    # -----------------------------------------------------------------------
    # forward
    #
    # Inference/evaluation entry point. Same signature as ContextTower.forward, so
    # your evaluate() can call it directly. Steps: prepare inputs -> encode -> return
    # only the citation vector (no LM loss, no gradients needed by the caller).
    # -----------------------------------------------------------------------
    def forward(self, input_ids, attention_mask):
        ids, mask, cite_pos = self.prepare(input_ids, attention_mask)   # marker removed, [CITE_VEC] appended
        return self._encode(ids, mask, cite_pos)[0]       # [0] = keep the vector, ignore the hidden states

    # -----------------------------------------------------------------------
    # forward_train
    #
    # Training entry point. Returns the citation vector AND (optionally) the language-modelling loss.
    #
    # LM loss = teacher forcing in ONE parallel pass: the model receives the full
    # ground-truth sequence, and the hidden state at position t is used to predict the
    # token at position t+1. Because of the causal mask, position t never sees t+1,
    # so there is no cheating, and every position gives a training signal at once.
    # Positions predicted: all context tokens plus [CITE_VEC] (the "stop and cite" signal).
    # Padding positions are excluded.
    #
    # with_lm=False skips the LM loss entirely (alignment-only ablation, saves memory).
    # -----------------------------------------------------------------------
    def forward_train(self, input_ids, attention_mask, with_lm: bool):
        ids, mask, cite_pos = self.prepare(input_ids, attention_mask)   # same preprocessing as forward()
        vec, h = self._encode(ids, mask, cite_pos)        # vec: [B, d] citation vectors, h: [B, L, H]

        lm_loss = None                                    # stays None when the LM loss is switched off
        if with_lm:
            # Teacher forcing in one parallel pass: hidden state at t predicts token t+1.
            # Targets are positions 1..cite_pos (context tokens + [CITE_VEC]); padding is excluded.
            L = ids.size(1)                               # padded sequence length
            # Boolean mask [B, L-1]: entry (b, t) is True when t < cite_pos[b].
            # "Source" position t predicts the token at t+1; t runs from 0 ([CLS]) up to
            # cite_pos-1 (last context token), so the targets are tokens 1..cite_pos.
            # arange(L-1)[None, :] is [1, L-1] and cite_pos[:, None] is [B, 1] -> broadcasts to [B, L-1].
            src_mask = torch.arange(L - 1, device=ids.device)[None, :] < cite_pos[:, None]
            h_src = h[:, :-1][src_mask]                   # [N_tokens, hidden]
            # h[:, :-1] drops the last position (nothing to predict after it); the boolean mask
            # then selects only the valid positions and flattens them into [N_tokens, H].
            tgt = ids[:, 1:][src_mask]                    # [N_tokens]
            # ids[:, 1:] = the "next token" for every position; same mask -> the true next tokens.
            logits = self.lm.cls(h_src)                   # vocab logits only where needed
            # self.lm.cls is the LM head (H -> V). Applying it ONLY on the selected positions
            # avoids building a huge [B, L, V] tensor.
            lm_loss = F.cross_entropy(logits.float(), tgt)   # mean next-token cross-entropy (in fp32 for stability)
        return vec, lm_loss                               # lm_loss is None if with_lm=False

    # -----------------------------------------------------------------------
    # get_param_groups
    #
    # Returns two parameter groups for the optimiser, each with its own learning rate:
    #   - the pretrained SciBERT (+ LM head): small lr (e.g. 1e-5) to avoid destroying
    #     what it already knows,
    #   - the freshly initialised citation head: larger lr (e.g. 1e-3) so it learns fast.
    # Same idea as ContextTower.get_param_groups.
    # -----------------------------------------------------------------------
    def get_param_groups(self, lr_backbone: float, lr_head: float) -> list:
        return [
            {"params": self.lm.parameters(), "lr": lr_backbone},      # SciBERT body + LM head
            {"params": list(self.proj.parameters()) + list(self.norm.parameters()), "lr": lr_head},   # citation head
        ]


# ---------------------------------------------------------------------------
# 3. Frozen paper embeddings from your trained Paper Module
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# load_frozen_paper_index
#
# Rebuilds your trained PaperTower from its checkpoint and runs it ONCE over all
# candidate papers to get the embedding matrix. These embeddings are the fixed
# "answer sheet" the citation vector must match: the PaperTower is never trained
# again here, so the target space cannot drift.
#
# Arguments: cfg (the config dict), device.
# Returns:
#   cand_embs : [N, d] CPU tensor, one embedding per candidate paper
#   cand_ids  : [N] global paper ids (row i of cand_embs belongs to cand_ids[i])
#   n_total   : size of the global paper-id space (used to size the id -> row lookup table)
# ---------------------------------------------------------------------------
def load_frozen_paper_index(cfg: dict, device):
    ckpt = torch.load(os.path.expanduser(cfg["paper_ckpt"]), map_location="cpu")   # load the saved two-tower checkpoint
    pcfg = ckpt["config"]                                 # the config that checkpoint was trained with (architecture hyperparameters)
    data_root = os.path.expanduser(cfg["data_root"])      # folder holding the feature / id files

    # Load the precomputed metapath feature matrices (one per metapath key, e.g. "P", "PP").
    # Row i = features of the paper whose global id is i.
    feats = {k: torch.load(os.path.join(data_root, f"feat_{k}.pt"), map_location="cpu")
             for k in pcfg["feat_keys"]}
    # Candidate pool = ALL papers of the ACL-200 KG (no corpus/external split), same as train_baseline.py.
    with open(os.path.join(data_root, "node_index.json"), encoding="utf-8") as f:
        n_papers = len(json.load(f)["paper"])
    cand_ids = torch.arange(n_papers, dtype=torch.long)

    # Rebuild the PaperTower with EXACTLY the hyperparameters it was trained with
    # (so the saved weights fit), then move it to the device.
    tower = PaperTower(
        feat_keys=pcfg["feat_keys"], nfeat=768, num_heads=pcfg["num_heads"],
        hidden=pcfg["hidden"], embed_dim=pcfg["embed_dim"], n_fp_layers=pcfg["n_fp_layers"],
        dropout=pcfg["dropout"], input_drop=pcfg["input_drop"], att_drop=pcfg["att_drop"],
        act=pcfg["act"], residual=pcfg["residual"], use_mlp=pcfg["use_mlp"],
    ).to(device)
    tower.load_state_dict(ckpt["paper_tower"])            # put the trained weights into it

    # Run the tower over every candidate (in chunks, in eval mode) -> [N, d] embeddings on CPU.
    cand_embs = build_candidate_index(tower, feats, cand_ids, device)       # [N, d] on CPU
    n_total = next(iter(feats.values())).size(0)                            # size of global-id space
    # (number of rows of any feature matrix = number of paper ids)

    del tower                                             # the tower is no longer needed...
    torch.cuda.empty_cache()                              # ...so free its GPU memory
    return cand_embs, cand_ids, n_total


# ---------------------------------------------------------------------------
# 4. One training epoch
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# train_one_epoch
#
# One full pass over the training data. For every batch:
#   1. find the target paper's row in the frozen embedding matrix,
#   2. run the model to get the citation vectors (+ LM loss),
#   3. alignment loss = cross-entropy over similarities to ALL frozen papers
#      (the correct paper is the target class),
#   4. total loss = alignment + lambda_lm * LM loss,
#   5. backward pass with mixed precision + gradient clipping + optimiser step.
#
# Arguments:
#   model, loader, optimizer, scaler : the usual training objects (scaler = AMP GradScaler)
#   train_embs : [N_kept, d] frozen paper embeddings on the GPU
#   id2row     : lookup table, global paper id -> row in train_embs (-1 = not in the pool)
#   device, cfg, epoch
# Returns: (dict with the mean total/alignment/LM loss, seconds the epoch took)
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, scaler, train_embs, id2row, device, cfg, epoch):
    model.train()                                         # enable dropout etc.
    lam, tau = cfg["lambda_lm"], cfg["temperature"]       # LM-loss weight and softmax temperature
    t0, n_batches = time.time(), 0                        # start the clock, batch counter
    tot, tot_align, tot_lm = 0.0, 0.0, 0.0                # running sums of the three losses

    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)   # progress bar wrapping the loader
    for batch in pbar:
        targets = id2row[batch["cited_paper_id"]]         # row in the frozen pool (-1 = not in pool)
        # batch["cited_paper_id"] holds global ids; id2row converts each to its row index
        # in train_embs. That row index is the "class label" of the example.
        valid = targets >= 0                              # True where the cited paper exists in the pool
        if not valid.any():                               # nothing usable in this batch
            continue                                      # skip it
        targets, valid = targets.to(device), valid.to(device)   # move labels to the GPU

        with torch.amp.autocast("cuda"):                  # mixed precision (fp16) for speed / memory
            vec, lm_loss = model.forward_train(
                batch["input_ids"].to(device), batch["attention_mask"].to(device),
                with_lm=lam > 0,                          # compute the LM loss only if its weight is > 0
            )

        # Alignment loss against the FULL frozen candidate pool (fp32)
        # [B, d] @ [d, N_kept] -> [B, N_kept]: similarity of each citation vector to every paper,
        # divided by the temperature (smaller tau = sharper softmax).
        logits = vec.float() @ train_embs.T / tau
        # Cross-entropy with the target row as the correct class: pushes the citation vector
        # towards its true paper and away from ALL other papers. [valid] drops unusable rows.
        align = F.cross_entropy(logits[valid], targets[valid])
        loss = align + (lam * lm_loss if lam > 0 else 0.0)   # combined objective

        if math.isnan(loss.item()) or math.isinf(loss.item()):   # corrupted batch?
            print(f"\n[WARN] NaN/Inf at epoch {epoch} batch {n_batches} - skipping.")
            optimizer.zero_grad()                         # discard any gradients
            n_batches += 1
            continue                                      # do NOT update the weights with garbage

        optimizer.zero_grad()                             # clear gradients of the previous step
        scaler.scale(loss).backward()                     # backward pass on the scaled loss (AMP needs scaling to avoid fp16 underflow)
        scaler.unscale_(optimizer)                        # undo the scaling so clipping sees real gradient sizes
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)   # cap the total gradient norm at 1.0 (stability)
        scaler.step(optimizer)                            # optimiser update (skipped automatically if grads are inf/nan)
        scaler.update()                                   # adjust the AMP scale factor for the next step

        tot += loss.item()                                # accumulate for the epoch averages
        tot_align += align.item()
        tot_lm += lm_loss.item() if lam > 0 else 0.0
        n_batches += 1
        pbar.set_postfix({"align": f"{align.item():.3f}",     # live loss display in the progress bar
                          "lm": f"{lm_loss.item():.3f}" if lam > 0 else "-"})

    n = max(n_batches, 1)                                 # avoid dividing by zero if no batch ran
    return {"loss": tot / n, "align": tot_align / n, "lm": tot_lm / n}, time.time() - t0


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# main
#
# Orchestrates everything:
#   1. read config, seed everything, create the output folder
#   2. build the frozen paper-embedding matrix from the trained Paper Module
#   3. build the CitationLM
#   4. build data loaders, then train with early stopping on validation MRR
#   5. reload the best checkpoint and evaluate once on the test set
#   6. save history.json
# ---------------------------------------------------------------------------
def main():
    # ----- command-line arguments -----
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)        # path to the JSON config (mandatory)
    args = parser.parse_args()
    cfg = load_config(args.config)                        # JSON -> dict

    # Print the config so every run is self-documented in the logs.
    print("=" * 60)
    print(f"  Experiment : {cfg['experiment_name']}")
    for k, v in cfg.items():
        print(f"    {k:20s} = {v}")
    print("=" * 60 + "\n")

    # ----- reproducibility: seed every random number generator -----
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"]); torch.cuda.manual_seed(cfg["seed"])

    # ----- device and folders -----
    device = torch.device(f"cuda:{cfg['gpu']}" if torch.cuda.is_available() else "cpu")   # chosen GPU, else CPU
    data_root = os.path.expanduser(cfg["data_root"])      # expand "~" to the home directory
    ckpt_dir = os.path.join(os.path.expanduser(cfg["output_dir"]), cfg["experiment_name"])   # where results go
    os.makedirs(ckpt_dir, exist_ok=True)                  # create it if missing
    with open(os.path.join(ckpt_dir, "config.json"), "w") as f:   # save a copy of the config next to the results
        json.dump(cfg, f, indent=2)

    # --- Frozen paper embeddings (computed once, never updated) ---
    print("Building frozen paper index from the trained Paper Module ...")
    cand_embs, cand_ids, n_total = load_frozen_paper_index(cfg, device)   # [N, d], [N], int
    embed_dim = cand_embs.size(1)                         # d, read from the checkpoint (so it can't mismatch)

    # Training pool = non-zero embeddings only (same filter evaluate() applies)
    # Papers with an all-zero embedding (e.g. no edges in the graph) are useless as targets/negatives.
    nonzero = cand_embs.abs().sum(dim=1) != 0             # [N] boolean: True if the embedding is not all zeros
    kept_ids = cand_ids[nonzero]                          # global ids of the papers we keep
    train_embs = cand_embs[nonzero].to(device)            # [N_kept, d]  frozen targets, on the GPU
    # Lookup table: global paper id -> row in train_embs. Filled with -1 ("not in the pool")...
    id2row = torch.full((n_total,), -1, dtype=torch.long)
    id2row[kept_ids] = torch.arange(len(kept_ids))        # ...then each kept id gets its row index
    print(f"Candidates: {len(cand_ids):,} total, {len(kept_ids):,} used for training "
          f"(embed_dim={embed_dim})\n")

    # --- Model ---
    model = CitationLM(
        model_name=cfg["scibert_model_name"],
        embed_dim=embed_dim,                              # output dimension = paper-embedding dimension
        dropout=cfg["dropout"],
        marker_strings=cfg["marker_strings"],
    ).to(device)
    print(f"CitationLM params: {sum(p.numel() for p in model.parameters()):,}\n")   # total parameter count

    # --- Data (identical splits to the ACL-200 two-tower baseline) ---
    datasets = build_datasets(                            # dict with "train" / "val" / "test" datasets
        all_contexts_path=os.path.join(data_root, "all_contexts.json"),
        node_index_path=os.path.join(data_root, "node_index.json"),
        max_length=cfg["max_length"],                     # token limit per context
        seed=cfg["seed"],
        split_path=os.path.join(data_root, "split_uris.json"),   # fixed train/val/test split
    )
    # Fail early with a clear message instead of a cryptic error deep inside a sampler.
    if len(datasets["train"]) == 0:
        raise RuntimeError("Empty train split - check that `dataset.py` comes from shared/acl200_prep "
                           "(not shared/data_prep) and that split_uris.json matches all_contexts.json.")
    # Standard shuffled loader: ACL-200 has no multi-citation grouping, so no sibling records
    # can end up in the same batch (no GroupAwareBatchSampler needed).
    train_loader = DataLoader(datasets["train"], batch_size=cfg["batch_size"], shuffle=True,
                              collate_fn=lcr_collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(datasets["val"], batch_size=cfg["batch_size"], shuffle=False,   # fixed order for evaluation
                            collate_fn=lcr_collate_fn, num_workers=4, pin_memory=True)
    test_loader = DataLoader(datasets["test"], batch_size=cfg["batch_size"], shuffle=False,
                             collate_fn=lcr_collate_fn, num_workers=4, pin_memory=True)

    # Sanity check: show one training context before/after prepare() to verify the citation marker is stripped.
    _b = next(iter(DataLoader(datasets["train"], batch_size=2, shuffle=False, collate_fn=lcr_collate_fn)))
    _ids, _m, _p = model.prepare(_b["input_ids"][:1].to(device), _b["attention_mask"][:1].to(device))
    print("RAW     :", model.tokenizer.decode(_b["input_ids"][0][_b["attention_mask"][0].bool()]))
    print("PREPARED:", model.tokenizer.decode(_ids[0][_m[0].bool()]), "\n")

    # Adam with two learning rates (backbone vs. citation head); scaler = mixed-precision gradient scaler.
    optimizer = torch.optim.Adam(model.get_param_groups(cfg["lr_backbone"], cfg["lr_head"]))
    scaler = torch.amp.GradScaler("cuda")

    # --- Training loop ---
    best_mrr, best_epoch, patience_ctr = 0.0, 0, 0        # early-stopping bookkeeping
    history, t_start = [], time.time()                    # per-epoch records, start time

    for epoch in range(1, cfg["epochs"] + 1):
        losses, secs = train_one_epoch(model, train_loader, optimizer, scaler,
                                       train_embs, id2row, device, cfg, epoch)   # one training pass
        metrics = evaluate(model, val_loader, cand_embs, cand_ids, device)       # rank all papers for every val context

        print(f"Epoch {epoch:3d} | loss={losses['loss']:.4f} (align={losses['align']:.4f} "
              f"lm={losses['lm']:.4f}) | time={secs:.1f}s"
              f" | R@1={metrics.get('Recall@1', 0):.4f} R@10={metrics.get('Recall@10', 0):.4f}"
              f" MRR={metrics.get('MRR', 0):.4f} nDCG@10={metrics.get('nDCG@10', 0):.4f}")
        history.append({"epoch": epoch, **losses, "epoch_secs": round(secs, 2), **metrics})   # keep for history.json

        if metrics.get("MRR", 0.0) > best_mrr:            # new best validation MRR?
            best_mrr, best_epoch, patience_ctr = metrics["MRR"], epoch, 0   # remember it, reset patience
            torch.save({"epoch": epoch, "model": model.state_dict(), "val_mrr": best_mrr,
                        "metrics": metrics, "config": cfg},
                       os.path.join(ckpt_dir, "best_model.pt"))   # checkpoint = best model so far
            print(f"  New best MRR={best_mrr:.4f} - saved")
        else:
            patience_ctr += 1                             # no improvement this epoch
            print(f"  (patience {patience_ctr}/{cfg['patience']})")
            if patience_ctr >= cfg["patience"]:           # too many epochs without improvement
                print(f"\nEarly stopping - best MRR={best_mrr:.4f} at epoch {best_epoch}")
                break                                     # stop training

    # --- Final test evaluation with the best checkpoint ---
    state = torch.load(os.path.join(ckpt_dir, "best_model.pt"), map_location=device)   # reload the BEST (not the last) model
    model.load_state_dict(state["model"])
    test_metrics = evaluate(model, test_loader, cand_embs, cand_ids, device)   # the test set is used only once, here
    print(f"\nTEST (epoch {best_epoch}): " +
          " ".join(f"{k}={v:.4f}" for k, v in test_metrics.items() if isinstance(v, float)))

    # Save everything about the run (curves + final test numbers).
    with open(os.path.join(ckpt_dir, "history.json"), "w") as f:
        json.dump({"best_mrr": best_mrr, "best_epoch": best_epoch,
                   "total_training_time_h": round((time.time() - t_start) / 3600, 4),
                   "history": history, "test_metrics": test_metrics}, f, indent=2)


if __name__ == "__main__":      # run main() only when executed as a script (not when imported)
    main()