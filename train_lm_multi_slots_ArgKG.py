"""
train_cite_lm_slots.py
----------------------
Autoregressive citation-vector LM with SEVERAL citation slots per context (Option B).

The model reads a citation context left-to-right (causal LM, teacher forcing), then emits one
[CITE_VEC] slot PER CITED PAPER. Each slot's hidden state is projected to a vector in the SAME
space as the frozen PaperTower embeddings; retrieval = nearest neighbour over those embeddings.

Sequence layout (citation marker moved to the END of the context, n = number of cited papers):
    [CLS] w1 w2 ... wm [CITE_VEC] x n  [CITE_END]

What the model reads at each position:
    context tokens   -> LM head: next word                                   (context LM loss)
    last context tok -> LM head: "[CITE_VEC]"  (at least one citation)        (count loss)
    slot i           -> vector head: the paper for this slot                  (alignment loss)
                     -> LM head: another [CITE_VEC] or [CITE_END]             (count loss)

Losses:
    alignment : each slot vector vs the FULL frozen candidate pool (InfoNCE). The gold papers of a
                context are an UNORDERED set, so slots are matched to gold papers one-to-one with
                the Hungarian algorithm (cosine cost) and the loss is applied to the matched pairs.
    LM        : next-token cross-entropy on the context tokens only (regulariser, can be switched off).
    count     : next-token cross-entropy on the structural tokens ([CITE_VEC] / [CITE_END]); this is
                how the model learns HOW MANY citations a context deserves.
    total = alignment + lambda_count * count + lambda_lm * LM

Inference needs NO generation loop: the slot token is always the same token, so one forward pass with
max_slots slots appended gives exactly what a step-by-step loop would give (the model is causal, so
slot i never sees slots after it). The number of slots used = 1 + number of consecutive slots whose
P([CITE_VEC] vs [CITE_END]) exceeds `count_threshold`.

Reuses from your two-tower training script:
    load_config, build_candidate_index, PaperTower, build_datasets, lcr_collate_fn
Needs in ~/HGNN/shared/data_prep:
    whole_group_sampler.py   (WholeGroupBatchSampler, batches that keep a context group together)

Usage:
    python train_cite_lm_slots.py --config cite_lm_slots_config.json
"""

# ===========================================================================
# HOW TO READ THIS FILE
# ---------------------------------------------------------------------------
#   1. Imports
#   2. Data helpers        - collapsing records into groups, pool de-duplication, slot<->paper matching,
#                            metric aggregation (the whole-group batch sampler is imported, see below)
#   3. CitationLM          - the model (input layout, slot vectors, losses, inference)
#   4. load_frozen_paper_index  - builds the fixed paper-embedding "answer sheet"   (unchanged)
#   5. Training            - matched alignment loss + one training epoch
#   6. Evaluation          - group-level ranking with automatic slot count
#   7. main                - wires everything together (setup -> train -> test)
#
# Shape notation used in the comments:
#   G = number of context GROUPS in a batch, R = number of records in a batch (R >= G),
#   L = sequence length, H = hidden size (768), d = embed_dim (paper-embedding size),
#   K = max_slots, Kg = gold papers per group (padded), N = candidate papers, V = vocabulary size.
# ===========================================================================

# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------
import os                          # file paths and folder creation
import json                        # saving config / history files
import math                        # isnan / isinf guard, log2 for nDCG, ceil for sampler length
import time                        # timing epochs and the whole run
import random                      # python RNG (seeding, shuffling groups)
import argparse                    # command-line arguments (--config)
import itertools                   # permutations (fallback when scipy is missing)
from collections import defaultdict   # dict that creates empty lists on demand

import numpy as np                 # numpy RNG seed, de-duplication, cost matrices
import torch                       # tensors, autograd, GPU
import torch.nn as nn              # layers (Linear, LayerNorm, Dropout) and nn.Module
import torch.nn.functional as F    # cross_entropy, normalize, softmax
from torch.utils.data import DataLoader   # batches a dataset
from tqdm import tqdm              # progress bars
from transformers import AutoTokenizer, BertLMHeadModel
# AutoTokenizer  : SciBERT tokenizer (text <-> token ids)
# BertLMHeadModel: BERT + LM head; with is_decoder=True it uses a CAUSAL attention mask

try:                                                          # scipy gives the Hungarian algorithm
    from scipy.optimize import linear_sum_assignment as _linear_sum_assignment
except ImportError:                                           # no scipy: a brute-force fallback is used
    _linear_sum_assignment = None

# Importing this module also does the sys.path setup of your original script.
# (evaluate and GroupAwareBatchSampler are no longer needed: this file has its own
#  group-level evaluation, and the whole-group batch sampler is imported from data_prep below.)
from train_domain_specific import (
    load_config,              # reads the JSON config file into a dict (new keys are fine)
    build_candidate_index,    # runs PaperTower over all candidate papers -> [N, d] embedding matrix
    PaperTower,               # the paper encoder you already trained
    build_datasets,           # builds train/val/test datasets of (context, cited paper) records
    lcr_collate_fn,           # turns a list of records into one batch of tensors
)
# Lives next to dataset.py / group_aware_sampler.py in ~/HGNN/shared/data_prep. This import MUST stay
# after the train_domain_specific import above: that import adds shared/data_prep to sys.path.
from whole_group_sampler import WholeGroupBatchSampler   # batches that keep all records of a group together


# ---------------------------------------------------------------------------
# 2. Data helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# collapse_groups
#
# Turns a batch of RECORDS (one row per cited paper) into a batch of GROUPS (one row per context).
# Records of the same group share the same context text, so we keep the first record's tokens and
# gather the cited papers of all its records into a padded list of pool rows.
#
# Arguments:
#   batch    : dict from lcr_collate_fn (input_ids, attention_mask, cited_paper_id, context_group_id)
#   id2row   : CPU LongTensor, global paper id -> row in the pool (-1 = not in the pool)
#   max_gold : keep at most this many gold papers per group (training: max_slots; eval: None = all)
# Returns None if no group has a usable gold paper, else a dict:
#   group_ids [G] (python list), input_ids [G, L], attention_mask [G, L],
#   gold_rows [G, Kg] (pool rows, padded with -1), n_gold [G] (number of gold papers)
# Gold papers that map to the SAME pool row (identical embeddings after de-duplication) count once.
# ---------------------------------------------------------------------------
def collapse_groups(batch, id2row, max_gold=None):
    group_ids = batch["context_group_id"].tolist()            # group id of every record
    cited_rows = id2row[batch["cited_paper_id"]].tolist()     # global paper id -> pool row, per record
    first, golds = {}, {}                                     # first record index / gold rows, per group
    for r, (g, row) in enumerate(zip(group_ids, cited_rows)): # walk over the records of the batch
        if g not in first:                                    # first time we see this group...
            first[g] = r                                      # ...remember which record carries its tokens
            golds[g] = []                                     # ...and start its list of gold rows
        if row >= 0 and row not in golds[g]:                  # usable paper, not a duplicate of an earlier gold
            golds[g].append(row)                              # add it to the group's gold set
    keep = [g for g in first if golds[g]]                     # groups that have at least one gold paper
    if not keep:                                              # nothing usable in this batch
        return None
    if max_gold is not None:                                  # cap the gold set (training only)
        golds = {g: v[:max_gold] for g, v in golds.items()}
    first_idx = torch.tensor([first[g] for g in keep], dtype=torch.long)   # which record to take tokens from
    kg = max(len(golds[g]) for g in keep)                     # widest gold list in this batch
    gold_rows = torch.full((len(keep), kg), -1, dtype=torch.long)          # [G, Kg] padded with -1
    for i, g in enumerate(keep):                              # fill each group's gold rows
        gold_rows[i, : len(golds[g])] = torch.tensor(golds[g])
    return {
        "group_ids": keep,                                    # python list of group ids
        "input_ids": batch["input_ids"][first_idx],           # [G, L] one context per group
        "attention_mask": batch["attention_mask"][first_idx], # [G, L]
        "gold_rows": gold_rows,                               # [G, Kg]
        "n_gold": (gold_rows >= 0).sum(dim=1),                # [G] number of gold papers per group
    }


# ---------------------------------------------------------------------------
# dedup_embeddings
#
# Collapses papers whose embeddings are identical (after rounding) into ONE row. With a full-pool
# softmax, a cited paper that has thousands of identical "twins" can never win against them, so its
# loss could not drop below ~log(cluster size). After de-duplication the target is the cluster.
# Arguments: E [N, d] numpy; decimals = rounding used to decide "identical".
# Returns: reps [M, d] (one real embedding per cluster), cluster_of_row [N] (row of E -> row of reps).
# ---------------------------------------------------------------------------
def dedup_embeddings(E, decimals=4):
    R = np.round(E, decimals) + 0.0                           # round; "+ 0.0" turns -0.0 into 0.0 so they match
    _, first_idx, inverse = np.unique(R, axis=0, return_index=True, return_inverse=True)   # group identical rows
    return E[first_idx], np.asarray(inverse).reshape(-1)      # one real embedding per cluster + row -> cluster map


# ---------------------------------------------------------------------------
# assign_slots
#
# One-to-one assignment between n slots and n gold papers that MINIMISES the total cost
# (cost[i][j] = -cosine(slot i, gold j)). Hungarian algorithm via scipy; if scipy is missing, brute
# force over all permutations (fine for n <= 4, slow for large n).
# Returns two lists (slot_idx, gold_idx): slot slot_idx[k] is assigned to gold gold_idx[k].
# ---------------------------------------------------------------------------
def assign_slots(cost):
    n = cost.shape[0]                                         # number of slots (= number of gold papers)
    if n == 1:                                                # trivial case: the only slot gets the only gold
        return [0], [0]
    if _linear_sum_assignment is not None:                    # preferred: Hungarian algorithm
        rows, cols = _linear_sum_assignment(cost)
        return rows.tolist(), cols.tolist()
    best_total, best_perm = None, None                        # fallback: try every permutation
    for perm in itertools.permutations(range(n)):
        total = sum(cost[i, perm[i]] for i in range(n))       # total cost of assigning slot i -> gold perm[i]
        if best_total is None or total < best_total:          # keep the cheapest assignment
            best_total, best_perm = total, perm
    return list(range(n)), list(best_perm)


# ---------------------------------------------------------------------------
# ranks_to_metrics
#
# Same macro-averaged metrics as the evaluate() of your two-tower script, given
# group_ranks[group_id] = list of ranks (one per gold paper).
# Recall@k = hits / min(n_papers, k) (not capped, exactly like evaluate()), MRR and nDCG@10-style
# discount averaged inside the group first, then across groups.
# Returns {} if there are no groups.
# ---------------------------------------------------------------------------
def ranks_to_metrics(group_ranks):
    k_values = [1, 5, 10, 20]                                 # same cutoffs as evaluate()
    recall_hits = {k: 0.0 for k in k_values}                  # running sums, one per k
    mrr_sum, ndcg_sum, n_queries = 0.0, 0.0, 0
    for ranks in group_ranks.values():                        # one entry per context group
        n_papers = len(ranks)                                 # number of gold papers of this group
        for k in k_values:
            denom = min(n_papers, k)                          # you cannot hit more than k papers in the top k
            hits = sum(1 for r in ranks if r <= k)            # gold papers inside the top k
            recall_hits[k] += hits / denom
        mrr_sum += sum(1.0 / r for r in ranks) / n_papers     # mean reciprocal rank inside the group
        ndcg_sum += sum(1.0 / math.log2(r + 1) for r in ranks) / n_papers
        n_queries += 1                                        # each GROUP counts as one query
    if n_queries == 0:
        return {}
    metrics = {f"Recall@{k}": recall_hits[k] / n_queries for k in k_values}
    metrics.update({"MRR": mrr_sum / n_queries, "nDCG@10": ndcg_sum / n_queries, "n_queries": n_queries})
    return metrics


# ---------------------------------------------------------------------------
# decide_counts
#
# Turns the per-slot probabilities p_more [G, K] (P([CITE_VEC]) vs P([CITE_END]) read at each slot)
# into the number of slots to USE. Slot 1 is always used (every context has at least one citation);
# slot i+1 is added only if slots 1..i all said "more" (probability above the threshold).
# cumprod makes the "all previous slots said more" condition: it stays 1 until the first 0.
# Returns a LongTensor [G] with values in 1..K.
# ---------------------------------------------------------------------------
def decide_counts(p_more, threshold):
    keep_going = (p_more[:, :-1] > threshold).long()          # [G, K-1]: did slot i ask for another slot?
    still_going = torch.cumprod(keep_going, dim=1)            # [G, K-1]: 1 until the first "no", then 0
    return 1 + still_going.sum(dim=1)                         # [G]: 1 + number of consecutive "more" answers


# ---------------------------------------------------------------------------
# 3. Model
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# CitationLM (class)
#
# SciBERT used as a CAUSAL language model + a small head that turns the hidden state at each
# [CITE_VEC] slot into a vector in the paper-embedding space.
#
# Entry points:
#   - forward_train(...)  training: slot vectors + context-LM loss + count loss
#   - slot_forward(...)   inference: K slot vectors and the P(another slot) at each slot
#   - forward(...)        same signature as ContextTower, returns the FIRST slot's vector only
#                         (so your original evaluate() still works as a single-vector evaluation)
# ---------------------------------------------------------------------------
class CitationLM(nn.Module):
    """
    SciBERT initialised as a CAUSAL LM (is_decoder=True) + a citation-vector head applied at
    every [CITE_VEC] slot. Input layout:  [CLS] context [CITE_VEC] x n [CITE_END].
    """

    # -----------------------------------------------------------------------
    # __init__
    #
    # Builds the tokenizer (with TWO new special tokens), the marker patterns to strip, the causal
    # SciBERT LM and the vector head (Dropout -> Linear -> LayerNorm).
    # Arguments: model_name, embed_dim (= d), dropout, marker_strings (e.g. ["[cite]"]),
    #            max_slots (K: maximum number of citation slots per context).
    # NOTE: the vocabulary now grows by TWO tokens, so checkpoints of the single-slot model cannot
    # be loaded into this model (embedding shapes differ).
    # -----------------------------------------------------------------------
    def __init__(self, model_name: str, embed_dim: int, dropout: float, marker_strings: list,
                 max_slots: int):
        super().__init__()                                    # initialise nn.Module internals
        self.max_slots = max_slots                            # K

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)   # SciBERT tokenizer

        # Token-id patterns of the in-text markers to strip (e.g. "[cite]" -> "[", "cite", "]").
        # Computed BEFORE adding the new special tokens so nothing collides.
        self.marker_seqs = [
            self.tokenizer(m, add_special_tokens=False)["input_ids"] for m in marker_strings
        ]

        # Two new special tokens: [CITE_VEC] = one citation slot, [CITE_END] = "no more slots".
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["[CITE_VEC]", "[CITE_END]"]})
        self.cite_id = self.tokenizer.convert_tokens_to_ids("[CITE_VEC]")   # id of the slot token
        self.end_id = self.tokenizer.convert_tokens_to_ids("[CITE_END]")    # id of the stop token
        self.pad_id = self.tokenizer.pad_token_id             # padding id
        self.cls_id = self.tokenizer.cls_token_id             # [CLS]: first token, acts as "start of text"

        # BERT with an LM head and a CAUSAL mask (position t only sees positions <= t).
        self.lm = BertLMHeadModel.from_pretrained(model_name, is_decoder=True)
        self.lm.resize_token_embeddings(len(self.tokenizer))  # make room for the two new tokens

        hidden = self.lm.config.hidden_size                   # H (768 for SciBERT)
        self.dropout = nn.Dropout(dropout)                    # dropout before the vector head
        self.proj = nn.Linear(hidden, embed_dim)              # H -> d
        self.norm = nn.LayerNorm(embed_dim)                   # stabilise before L2 normalisation
        nn.init.xavier_uniform_(self.proj.weight, gain=nn.init.calculate_gain("relu"))   # as in ContextTower
        nn.init.zeros_(self.proj.bias)                        # zero bias

    # -----------------------------------------------------------------------
    # _strip_markers  (unchanged)
    #
    # Removes every occurrence of the marker token-patterns from a list of token ids. The citation
    # slots now live at the END of the context, so the original in-text marker must be deleted.
    # -----------------------------------------------------------------------
    def _strip_markers(self, ids: list) -> list:
        out, i = [], 0                                        # cleaned ids, scan position
        while i < len(ids):
            for seq in self.marker_seqs:                      # try every marker pattern at position i
                if ids[i : i + len(seq)] == seq:              # pattern starts here?
                    i += len(seq)                             # yes: skip the whole marker
                    break
            else:                                             # no pattern matched at this position:
                out.append(ids[i])                            # keep the token
                i += 1
        return out

    # -----------------------------------------------------------------------
    # prepare
    #
    # Converts tokenised contexts ([CLS] body [SEP] + padding) into the layout this model needs:
    #     [CLS] body_without_marker [CITE_VEC] x n_slots [CITE_END]      (append_end=True, training)
    #     [CLS] body_without_marker [CITE_VEC] x n_slots                 (append_end=False, inference)
    # n_slots is given PER ROW (training: the true number of gold papers; inference: always K).
    #
    # Returns (all on the input's device):
    #   ids [G, L], mask [G, L]       the padded batch
    #   slot_pos [G, K]               position of slot j in the sequence (0 where the slot is unused)
    #   slot_valid [G, K]             True where slot j exists (j < n_slots)
    #   slot_start [G]                index of the FIRST slot (= length of [CLS] + body)
    # Runs under no_grad: it only manipulates integer ids.
    # -----------------------------------------------------------------------
    @torch.no_grad()
    def prepare(self, input_ids, attention_mask, n_slots, append_end):
        device = input_ids.device                             # remember where to send the results
        K = self.max_slots                                    # maximum number of slots
        rows, slot_start = [], []                             # new token lists / index of first slot
        for ids, m, n in zip(input_ids.cpu(), attention_mask.cpu(), n_slots.tolist()):
            toks = ids[m.bool()].tolist()                     # real tokens only (drop padding)
            body = self._strip_markers(toks[1:-1])            # drop [CLS]/[SEP], strip the in-text markers
            prefix = [self.cls_id] + body                     # everything before the slots
            row = prefix + [self.cite_id] * n + ([self.end_id] if append_end else [])
            rows.append(row)                                  # full new sequence
            slot_start.append(len(prefix))                    # first slot sits right after the prefix

        L = max(len(r) for r in rows)                         # padded length of the batch
        new_ids = torch.full((len(rows), L), self.pad_id, dtype=torch.long)   # [G, L] filled with PAD
        new_mask = torch.zeros((len(rows), L), dtype=torch.long)              # [G, L] filled with 0
        for b, r in enumerate(rows):                          # copy each row into the padded tensors
            new_ids[b, : len(r)] = torch.tensor(r)
            new_mask[b, : len(r)] = 1                         # 1 = real token

        slot_start = torch.tensor(slot_start, dtype=torch.long)               # [G]
        j = torch.arange(K)[None, :]                          # [1, K] slot numbers 0..K-1
        slot_valid = j < n_slots.cpu()[:, None]               # [G, K] slot j exists if j < n_slots
        slot_pos = torch.where(slot_valid, slot_start[:, None] + j,           # position of slot j ...
                               torch.zeros_like(slot_valid, dtype=torch.long))  # ... or 0 if unused
        return (new_ids.to(device), new_mask.to(device), slot_pos.to(device),
                slot_valid.to(device), slot_start.to(device))

    # -----------------------------------------------------------------------
    # _encode
    #
    # Runs the causal SciBERT and reads the hidden state at every slot position, then turns each one
    # into an L2-normalised citation vector.
    # Returns: vecs [G, K, d] (unit vectors), h [G, L, H] (all hidden states), h_slots [G, K, H].
    # Entries of unused slots are meaningless (they read position 0); callers must mask them.
    # -----------------------------------------------------------------------
    def _encode(self, ids, mask, slot_pos):
        h = self.lm.bert(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state   # [G, L, H]
        b = torch.arange(h.size(0), device=h.device)[:, None]                  # [G, 1] row index
        h_slots = h[b, slot_pos]                              # [G, K, H]: hidden state at each slot position
        x = self.norm(self.proj(self.dropout(h_slots)))       # dropout -> Linear(H -> d) -> LayerNorm
        return F.normalize(x, p=2, dim=-1), h, h_slots        # unit length: dot product = cosine

    # -----------------------------------------------------------------------
    # slot_forward   (inference / evaluation)
    #
    # Appends max_slots slots to every context and runs ONE forward pass. Because the model is
    # causal and the slot token is always the same token, slot i's hidden state depends only on the
    # context and slots 1..i-1, which is exactly what a step-by-step generation loop would produce.
    # Returns:
    #   vecs   [G, K, d]  citation vector of every slot
    #   p_more [G, K]     at slot i: P([CITE_VEC]) among {[CITE_VEC], [CITE_END]} = "add another slot?"
    # -----------------------------------------------------------------------
    def slot_forward(self, input_ids, attention_mask):
        G = input_ids.size(0)                                 # number of contexts
        n_slots = torch.full((G,), self.max_slots, dtype=torch.long)           # always K slots
        ids, mask, slot_pos, _, _ = self.prepare(input_ids, attention_mask, n_slots, append_end=False)
        vecs, _, h_slots = self._encode(ids, mask, slot_pos)  # [G, K, d], ..., [G, K, H]
        two = self.lm.cls(h_slots)[..., [self.cite_id, self.end_id]].float()   # [G, K, 2] logits of the 2 tokens
        p_more = F.softmax(two, dim=-1)[..., 0]               # [G, K] softmax over the two -> P([CITE_VEC])
        return vecs, p_more

    # -----------------------------------------------------------------------
    # forward
    #
    # Same signature as ContextTower.forward. Returns only the FIRST slot's vector [G, d], so your
    # original evaluate() can still be run on this model as a single-vector evaluation.
    # -----------------------------------------------------------------------
    def forward(self, input_ids, attention_mask):
        return self.slot_forward(input_ids, attention_mask)[0][:, 0]

    # -----------------------------------------------------------------------
    # forward_train
    #
    # Training entry point. Feeds the TRUE sequence (teacher forcing, one parallel pass):
    #     [CLS] body [CITE_VEC] x n [CITE_END]
    # Hidden state at position t predicts the token at position t+1. Two groups of predictions:
    #   context LM : sources t = 0 .. slot_start-2  -> targets = the context tokens
    #   count      : sources t = slot_start-1 .. last-1 -> targets = [CITE_VEC] x n, [CITE_END]
    #                (the last context token must predict "a slot follows"; slot i predicts "another
    #                 slot" or "end"; this is how the model learns the number of citations)
    # with_lm=False skips the context-LM part entirely (cheaper) but still computes the count loss.
    # Arguments: n_slots = LongTensor [G] true number of gold papers per group (CPU).
    # Returns: vecs [G, K, d], lm_loss (scalar, 0 if with_lm=False), count_loss (scalar).
    # -----------------------------------------------------------------------
    def forward_train(self, input_ids, attention_mask, n_slots, with_lm: bool):
        ids, mask, slot_pos, _, slot_start = self.prepare(input_ids, attention_mask, n_slots, append_end=True)
        vecs, h, _ = self._encode(ids, mask, slot_pos)        # slot vectors + all hidden states

        L = ids.size(1)                                       # padded sequence length
        t = torch.arange(L - 1, device=ids.device)[None, :]   # [1, L-1] "source" positions
        last_pos = mask.sum(dim=1) - 1                        # [G] index of the last real token ([CITE_END])
        src_all = t < last_pos[:, None]                       # [G, L-1] positions that predict a real next token
        struct = src_all & (t >= (slot_start - 1)[:, None])   # [G, L-1] positions predicting slots / END
        sel = src_all if with_lm else struct                  # which sources we actually compute logits for
        h_src = h[:, :-1][sel]                                # [N_sel, H] hidden states at the selected sources
        tgt = ids[:, 1:][sel]                                 # [N_sel] the true next tokens
        is_struct = struct[sel]                               # [N_sel] True for count positions, False for context

        logits = self.lm.cls(h_src)                           # LM head only where needed: [N_sel, V]
        ce = F.cross_entropy(logits.float(), tgt, reduction="none")   # per-token loss (fp32)
        count_loss = ce[is_struct].mean()                     # every example has >= 2 count tokens
        if with_lm and (~is_struct).any():                    # context LM loss (needs at least one context token)
            lm_loss = ce[~is_struct].mean()
        else:
            lm_loss = ce.new_zeros(())                        # nothing to average: loss 0 (no gradient)
        return vecs, lm_loss, count_loss

    # -----------------------------------------------------------------------
    # get_param_groups  (unchanged)
    #
    # Two parameter groups with different learning rates: the pretrained SciBERT (+ LM head) gets a
    # small one, the freshly initialised vector head a larger one.
    # -----------------------------------------------------------------------
    def get_param_groups(self, lr_backbone: float, lr_head: float) -> list:
        return [
            {"params": self.lm.parameters(), "lr": lr_backbone},
            {"params": list(self.proj.parameters()) + list(self.norm.parameters()), "lr": lr_head},
        ]


# ---------------------------------------------------------------------------
# 4. Frozen paper embeddings from your trained Paper Module   (unchanged)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# load_frozen_paper_index
#
# Rebuilds your trained PaperTower from its checkpoint and runs it ONCE over all candidate papers.
# These embeddings are the fixed "answer sheet" the slot vectors must match.
# Returns: cand_embs [N, d] (CPU), cand_ids [N] (global ids), n_total (size of the global-id space).
# ---------------------------------------------------------------------------
def load_frozen_paper_index(cfg: dict, device):
    ckpt = torch.load(os.path.expanduser(cfg["paper_ckpt"]), map_location="cpu")   # saved two-tower checkpoint
    pcfg = ckpt["config"]                                     # hyperparameters it was trained with
    data_root = os.path.expanduser(cfg["data_root"])          # folder with feature / id files

    feats = {k: torch.load(os.path.join(data_root, f"feat_{k}.pt"), map_location="cpu")
             for k in pcfg["feat_keys"]}                      # metapath feature matrices, row i = paper id i
    cand_ids = torch.cat([                                    # candidate pool = corpus + external papers
        torch.load(os.path.join(data_root, "corpus_ids.pt"),   map_location="cpu"),
        torch.load(os.path.join(data_root, "external_ids.pt"), map_location="cpu"),
    ]).unique()                                               # remove duplicates (and sort)

    tower = PaperTower(                                       # rebuild with EXACTLY the training hyperparameters
        feat_keys=pcfg["feat_keys"], nfeat=768, num_heads=pcfg["num_heads"],
        hidden=pcfg["hidden"], embed_dim=pcfg["embed_dim"], n_fp_layers=pcfg["n_fp_layers"],
        dropout=pcfg["dropout"], input_drop=pcfg["input_drop"], att_drop=pcfg["att_drop"],
        act=pcfg["act"], residual=pcfg["residual"], use_mlp=pcfg["use_mlp"],
    ).to(device)
    tower.load_state_dict(ckpt["paper_tower"])                # trained weights

    cand_embs = build_candidate_index(tower, feats, cand_ids, device)   # [N, d] on CPU
    n_total = next(iter(feats.values())).size(0)              # number of paper ids

    del tower                                                 # free the GPU memory
    torch.cuda.empty_cache()
    return cand_embs, cand_ids, n_total


# ---------------------------------------------------------------------------
# 5. Training
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# slot_alignment_loss
#
# Alignment loss for a batch of groups. For every group:
#   1. cost[i][j] = -cosine(slot i, gold paper j)  (no gradient)
#   2. one-to-one assignment slots <-> gold papers (Hungarian algorithm)
# Then the usual InfoNCE (cross-entropy over the FULL frozen pool) is applied to every matched pair
# (slot vector, its assigned paper). Because the assignment is recomputed from the model's current
# outputs, two slots aiming at the same paper cannot both be matched to it, which pushes slots apart.
# Arguments:
#   vecs       [G, K, d]  slot vectors from forward_train
#   gold_rows  [G, Kg]    CPU LongTensor of gold pool rows, padded with -1 (n_gold of them per group)
#   train_embs [M, d]     frozen (de-duplicated) paper embeddings on the device
#   tau        softmax temperature
# Returns: (loss, number of matched pairs).
# ---------------------------------------------------------------------------
def slot_alignment_loss(vecs, gold_rows, train_embs, tau):
    dev = vecs.device                                         # where the vectors live
    pair_g, pair_i, pair_t = [], [], []                       # group index, slot index, target pool row
    gold_list = gold_rows.tolist()                            # python lists (no device syncs in the loop)
    with torch.no_grad():                                     # matching is a discrete decision: no gradient
        for g in range(vecs.size(0)):                         # for every group in the batch
            golds = [x for x in gold_list[g] if x >= 0]       # its real gold rows (drop the -1 padding)
            n = len(golds)                                    # number of slots used = number of gold papers
            q = vecs[g, :n].float()                           # [n, d] its slot vectors
            e = train_embs[torch.tensor(golds, device=dev)]   # [n, d] its gold embeddings
            cost = (-(q @ e.T)).cpu().numpy()                 # [n, n] negative cosine similarity
            slot_idx, gold_idx = assign_slots(cost)           # best one-to-one assignment
            for i, j in zip(slot_idx, gold_idx):              # record every matched pair
                pair_g.append(g)
                pair_i.append(i)
                pair_t.append(golds[j])
    pair_g = torch.tensor(pair_g, device=dev)                 # [P]
    pair_i = torch.tensor(pair_i, device=dev)                 # [P]
    pair_t = torch.tensor(pair_t, device=dev)                 # [P] target rows (class labels)
    q_sel = vecs[pair_g, pair_i].float()                      # [P, d] the matched slot vectors (WITH gradient)
    logits = q_sel @ train_embs.T / tau                       # [P, M] similarity to every frozen paper
    return F.cross_entropy(logits, pair_t), len(pair_t)       # mean InfoNCE over all matched pairs


# ---------------------------------------------------------------------------
# train_one_epoch
#
# One pass over the training data. For every batch of context groups:
#   1. collapse records into groups (one example per context, several gold papers),
#   2. teacher-forced forward pass -> slot vectors, context-LM loss, count loss,
#   3. matched alignment loss,
#   4. total = alignment + lambda_count * count + lambda_lm * LM,
#   5. backward pass with mixed precision, gradient clipping, optimiser step.
# Arguments: model, loader, optimizer, scaler (AMP GradScaler), train_embs [M, d] on the device,
#            id2row (global paper id -> row of train_embs), device, cfg, epoch.
# Returns: (dict with mean loss / align / lm / count, seconds the epoch took).
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, scaler, train_embs, id2row, device, cfg, epoch):
    model.train()                                             # enable dropout
    lam_lm, lam_cnt, tau = cfg["lambda_lm"], cfg["lambda_count"], cfg["temperature"]
    K = cfg["max_slots"]                                      # gold papers beyond K are ignored in training
    t0, n_batches = time.time(), 0                            # clock and batch counter
    tot, tot_align, tot_lm, tot_cnt = 0.0, 0.0, 0.0, 0.0      # running sums of the losses

    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)   # progress bar
    for batch in pbar:
        if cfg["debug_batches"] and n_batches >= cfg["debug_batches"]:   # smoke-test mode: stop early
            break
        grp = collapse_groups(batch, id2row, max_gold=K)      # records -> groups with gold sets
        if grp is None:                                       # nothing usable in this batch
            continue

        with torch.amp.autocast("cuda"):                      # mixed precision for speed / memory
            vecs, lm_loss, count_loss = model.forward_train(
                grp["input_ids"].to(device), grp["attention_mask"].to(device),
                grp["n_gold"],                                # one slot per gold paper (teacher forcing)
                with_lm=lam_lm > 0,                           # skip the context-LM part when its weight is 0
            )

        align, n_pairs = slot_alignment_loss(vecs, grp["gold_rows"], train_embs, tau)   # matched InfoNCE (fp32)
        loss = align + lam_cnt * count_loss.float() + (lam_lm * lm_loss.float() if lam_lm > 0 else 0.0)

        if math.isnan(loss.item()) or math.isinf(loss.item()):   # corrupted batch?
            print(f"\n[WARN] NaN/Inf at epoch {epoch} batch {n_batches} - skipping.")
            optimizer.zero_grad()
            n_batches += 1
            continue

        optimizer.zero_grad()                                 # clear old gradients
        scaler.scale(loss).backward()                         # backward pass on the scaled loss
        scaler.unscale_(optimizer)                            # undo scaling so clipping sees real sizes
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)   # cap gradient norm (stability)
        scaler.step(optimizer)                                # optimiser update
        scaler.update()                                       # adapt the AMP scale factor

        tot += loss.item()                                    # accumulate epoch averages
        tot_align += align.item()
        tot_lm += lm_loss.item() if lam_lm > 0 else 0.0
        tot_cnt += count_loss.item()
        n_batches += 1
        pbar.set_postfix({"align": f"{align.item():.3f}", "count": f"{count_loss.item():.3f}",
                          "lm": f"{lm_loss.item():.3f}" if lam_lm > 0 else "-"})

    n = max(n_batches, 1)                                     # avoid dividing by zero
    return ({"loss": tot / n, "align": tot_align / n, "lm": tot_lm / n, "count": tot_cnt / n},
            time.time() - t0)


# ---------------------------------------------------------------------------
# 6. Evaluation
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# build_eval_pool
#
# Builds the evaluation pool exactly like your evaluate() does: drop all-zero embeddings, keep the
# remaining ones (NOT de-duplicated, so numbers stay comparable with the baseline).
# Returns: E [N_pool, d] on the device, id2pool (CPU LongTensor global id -> pool row, -1 = absent),
#          number of zero-vector candidates dropped.
# ---------------------------------------------------------------------------
def build_eval_pool(cand_embs, cand_ids, n_total, device):
    nonzero = cand_embs.abs().sum(dim=1) != 0                 # True where the embedding is not all zeros
    E = cand_embs[nonzero].to(device)                         # filtered pool on the device
    kept = cand_ids[nonzero]                                  # global ids of the kept papers
    id2pool = torch.full((n_total,), -1, dtype=torch.long)    # -1 = "not in the pool"
    id2pool[kept] = torch.arange(len(kept))                   # kept papers get their row index
    return E, id2pool, int((~nonzero).sum())


# ---------------------------------------------------------------------------
# evaluate_slots
#
# Group-level evaluation of the multi-slot model with the same metrics as your evaluate().
# For every context group the model gives up to K slot vectors; a paper's score is its MAXIMUM
# cosine over the slots that are used, and a gold paper's rank = 1 + number of papers scoring
# strictly higher (same tie rule as evaluate()). Three variants are scored from the same vectors:
#   auto   : slots used = automatic count (decide_counts)     -> headline metrics (MRR, Recall@k ...)
#   slot1  : only the first slot                               -> "single vector" reference
#   oracle : slots used = true number of gold papers           -> upper bound for the count decision
# Extra outputs: count accuracy, mean predicted/true count, and the same variants restricted to
# MULTI-citation groups (prefix "multi_").
# Arguments: model, loader (whole-group batches), eval_pool (from build_eval_pool), device, cfg.
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_slots(model, loader, eval_pool, device, cfg):
    model.eval()                                              # no dropout
    E, id2pool, n_zero = eval_pool                            # pool embeddings, id -> row map, dropped count
    K, thr = cfg["max_slots"], cfg["count_threshold"]         # slot cap and "add another slot" threshold
    variants = ("auto", "slot1", "oracle")
    group_ranks = {v: defaultdict(list) for v in variants}    # variant -> group id -> ranks of its gold papers
    n_groups = n_exact = 0                                    # count statistics
    pred_sum = true_sum = 0.0

    for bi, batch in enumerate(tqdm(loader, desc="  Evaluating", leave=False)):
        if cfg["debug_batches"] and bi >= cfg["debug_batches"]:   # smoke-test mode: stop early
            break
        grp = collapse_groups(batch, id2pool, max_gold=None)  # keep ALL gold papers of every group
        if grp is None:
            continue
        vecs, p_more = model.slot_forward(grp["input_ids"].to(device), grp["attention_mask"].to(device))
        n_hat = decide_counts(p_more, thr)                    # [G] automatic number of slots
        n_true = grp["n_gold"].to(device).clamp(max=K)        # [G] true count (capped at K slots)
        n_used = {"auto": n_hat, "slot1": torch.ones_like(n_hat), "oracle": n_true}

        all_sims = torch.einsum("gkd,nd->gkn", vecs.float(), E)   # [G, K, N] cosine of every slot to every paper
        gold_rows = grp["gold_rows"].to(device)               # [G, Kg] gold pool rows (-1 padded)
        rows_c = gold_rows.clamp(min=0)                       # padded entries point at row 0 (ignored below)
        valid_list = (gold_rows >= 0).cpu().tolist()          # which gold entries are real

        for v in variants:
            slot_ok = torch.arange(K, device=device)[None, :] < n_used[v][:, None]   # [G, K] slots in use
            fused = all_sims.masked_fill(~slot_ok[:, :, None], float("-inf")).max(dim=1).values   # [G, N] max over used slots
            s_gold = fused.gather(1, rows_c)                  # [G, Kg] score of each gold paper
            ahead = (fused[:, None, :] > s_gold[:, :, None]).sum(dim=2)   # [G, Kg] papers scoring strictly higher
            ranks = (ahead + 1).cpu().tolist()                # 1-indexed ranks
            for g, gid in enumerate(grp["group_ids"]):        # store the ranks under their group
                for j, ok in enumerate(valid_list[g]):
                    if ok:
                        group_ranks[v][gid].append(ranks[g][j])

        n_groups += len(grp["group_ids"])                     # count statistics for the "auto" variant
        n_exact += int((n_hat == n_true).sum())
        pred_sum += float(n_hat.sum())
        true_sum += float(n_true.sum())

    metrics = ranks_to_metrics(group_ranks["auto"])           # headline metrics = automatic slot count
    if not metrics:                                           # nothing was evaluated (e.g. empty loader)
        return {"MRR": 0.0}
    metrics["n_zero_candidates_dropped"] = n_zero
    for v in ("slot1", "oracle"):                             # reference variants
        m = ranks_to_metrics(group_ranks[v])
        metrics[f"{v}_MRR"] = m.get("MRR", 0.0)
        metrics[f"{v}_Recall@10"] = m.get("Recall@10", 0.0)
    for v in variants:                                        # same variants on multi-citation groups only
        multi = {g: r for g, r in group_ranks[v].items() if len(r) >= 2}
        m = ranks_to_metrics(multi)
        metrics[f"multi_{v}_MRR"] = m.get("MRR", 0.0)
        metrics[f"multi_{v}_Recall@10"] = m.get("Recall@10", 0.0)
        metrics["multi_n_queries"] = float(len(multi)) if v == "auto" else metrics.get("multi_n_queries", 0.0)
    metrics["count_acc"] = n_exact / max(n_groups, 1)         # share of groups with exactly the right count
    metrics["pred_count_mean"] = pred_sum / max(n_groups, 1)  # average predicted number of slots
    metrics["true_count_mean"] = true_sum / max(n_groups, 1)  # average true number of slots
    return metrics


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# main
#
#   1. read config, seed everything, create the output folder
#   2. build the frozen paper-embedding matrix (+ de-duplicated training pool, + evaluation pool)
#   3. build the CitationLM
#   4. build whole-group data loaders, train with early stopping on validation MRR (automatic count)
#   5. reload the best checkpoint and evaluate once on the test set
#   6. save history.json
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)            # path to the JSON config
    args = parser.parse_args()
    cfg = load_config(args.config)                            # JSON -> dict

    print("=" * 60)                                           # print the config: every run is self-documented
    print(f"  Experiment : {cfg['experiment_name']}")
    for k, v in cfg.items():
        print(f"    {k:20s} = {v}")
    print("=" * 60 + "\n")

    random.seed(cfg["seed"]); np.random.seed(cfg["seed"])    # reproducibility
    torch.manual_seed(cfg["seed"]); torch.cuda.manual_seed(cfg["seed"])

    device = torch.device(f"cuda:{cfg['gpu']}" if torch.cuda.is_available() else "cpu")
    data_root = os.path.expanduser(cfg["data_root"])
    ckpt_dir = os.path.join(os.path.expanduser(cfg["output_dir"]), cfg["experiment_name"])
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(ckpt_dir, "config.json"), "w") as f:   # keep a copy of the config with the results
        json.dump(cfg, f, indent=2)

    # --- Frozen paper embeddings (computed once, never updated) ---
    print("Building frozen paper index from the trained Paper Module ...")
    cand_embs, cand_ids, n_total = load_frozen_paper_index(cfg, device)   # [N, d], [N], int
    embed_dim = cand_embs.size(1)                             # d, read from the checkpoint

    # Training pool = non-zero embeddings (same filter as evaluation), optionally de-duplicated.
    nonzero = cand_embs.abs().sum(dim=1) != 0                 # drop all-zero embeddings
    kept_ids = cand_ids[nonzero]                              # global ids of the kept papers
    E_np = cand_embs[nonzero].numpy().astype(np.float32)      # [N_kept, d]
    if cfg["dedup_pool"]:                                     # collapse identical embeddings into one target row
        reps, cluster_of_row = dedup_embeddings(E_np)
        train_embs = torch.from_numpy(reps).to(device)        # [M, d] one row per distinct embedding
        row_of_kept = torch.from_numpy(cluster_of_row).long() # kept paper -> its cluster row
    else:
        train_embs = torch.from_numpy(E_np).to(device)        # [N_kept, d]
        row_of_kept = torch.arange(len(kept_ids))             # every paper is its own row
    id2row = torch.full((n_total,), -1, dtype=torch.long)     # global paper id -> training-pool row (-1 = absent)
    id2row[kept_ids] = row_of_kept
    eval_pool = build_eval_pool(cand_embs, cand_ids, n_total, device)   # pool used for ranking at evaluation
    print(f"Candidates: {len(cand_ids):,} total, {len(kept_ids):,} non-zero, "
          f"{len(train_embs):,} distinct training targets (embed_dim={embed_dim})\n")

    # --- Model ---
    model = CitationLM(
        model_name=cfg["scibert_model_name"],
        embed_dim=embed_dim,                                  # output dimension = paper-embedding dimension
        dropout=cfg["dropout"],
        marker_strings=cfg["marker_strings"],
        max_slots=cfg["max_slots"],
    ).to(device)
    print(f"CitationLM params: {sum(p.numel() for p in model.parameters()):,}\n")

    # --- Data: whole-group batches (all papers cited by one marker travel together) ---
    datasets = build_datasets(
        all_contexts_path=os.path.join(data_root, "all_contexts.json"),
        node_index_path=os.path.join(data_root, "node_index.json"),
        max_length=cfg["max_length"],
        seed=cfg["seed"],
        split_path=os.path.join(data_root, "split_uris.json"),
    )

    def make_loader(split, shuffle):                          # one loader + its sampler per split
        sampler = WholeGroupBatchSampler(
            group_ids=[r.context_group_id for r in datasets[split].records],
            groups_per_batch=cfg["batch_size"],               # NOTE: batch_size now counts context GROUPS
            seed=cfg["seed"], shuffle=shuffle)
        loader = DataLoader(datasets[split], batch_sampler=sampler, collate_fn=lcr_collate_fn,
                            num_workers=4, pin_memory=True)
        return loader, sampler

    train_loader, train_sampler = make_loader("train", shuffle=True)    # reshuffled every epoch
    val_loader, _ = make_loader("val", shuffle=False)                   # fixed order
    test_loader, _ = make_loader("test", shuffle=False)

    optimizer = torch.optim.Adam(model.get_param_groups(cfg["lr_backbone"], cfg["lr_head"]))
    scaler = torch.amp.GradScaler("cuda")

    # --- Training loop ---
    best_mrr, best_epoch, patience_ctr = 0.0, 0, 0            # early-stopping bookkeeping
    history, t_start = [], time.time()

    for epoch in range(1, cfg["epochs"] + 1):
        train_sampler.set_epoch(epoch)                        # new group order each epoch

        losses, secs = train_one_epoch(model, train_loader, optimizer, scaler,
                                       train_embs, id2row, device, cfg, epoch)
        metrics = evaluate_slots(model, val_loader, eval_pool, device, cfg)    # group-level validation

        print(f"Epoch {epoch:3d} | loss={losses['loss']:.4f} (align={losses['align']:.4f} "
              f"lm={losses['lm']:.4f} count={losses['count']:.4f}) | time={secs:.1f}s")
        print(f"   VAL auto: R@1={metrics.get('Recall@1', 0):.4f} R@10={metrics.get('Recall@10', 0):.4f} "
              f"MRR={metrics.get('MRR', 0):.4f} | slot1 MRR={metrics.get('slot1_MRR', 0):.4f} "
              f"oracle-count MRR={metrics.get('oracle_MRR', 0):.4f}")
        print(f"   multi-citation groups ({metrics.get('multi_n_queries', 0):.0f}): "
              f"MRR auto={metrics.get('multi_auto_MRR', 0):.4f} slot1={metrics.get('multi_slot1_MRR', 0):.4f} "
              f"oracle={metrics.get('multi_oracle_MRR', 0):.4f} | count acc={metrics.get('count_acc', 0):.3f} "
              f"(pred mean {metrics.get('pred_count_mean', 0):.2f} vs true {metrics.get('true_count_mean', 0):.2f})")
        history.append({"epoch": epoch, **losses, "epoch_secs": round(secs, 2), **metrics})

        if metrics.get("MRR", 0.0) > best_mrr:                # new best validation MRR (automatic count)
            best_mrr, best_epoch, patience_ctr = metrics["MRR"], epoch, 0
            torch.save({"epoch": epoch, "model": model.state_dict(), "val_mrr": best_mrr,
                        "metrics": metrics, "config": cfg},
                       os.path.join(ckpt_dir, "best_model.pt"))
            print(f"  New best MRR={best_mrr:.4f} - saved")
        else:
            patience_ctr += 1
            print(f"  (patience {patience_ctr}/{cfg['patience']})")
            if patience_ctr >= cfg["patience"]:
                print(f"\nEarly stopping - best MRR={best_mrr:.4f} at epoch {best_epoch}")
                break

    # --- Final test evaluation with the best checkpoint ---
    state = torch.load(os.path.join(ckpt_dir, "best_model.pt"), map_location=device)
    model.load_state_dict(state["model"])                     # reload the BEST (not the last) model
    test_metrics = evaluate_slots(model, test_loader, eval_pool, device, cfg)
    print(f"\nTEST (epoch {best_epoch}): " +
          " ".join(f"{k}={v:.4f}" for k, v in test_metrics.items() if isinstance(v, float)))

    with open(os.path.join(ckpt_dir, "history.json"), "w") as f:
        json.dump({"best_mrr": best_mrr, "best_epoch": best_epoch,
                   "total_training_time_h": round((time.time() - t_start) / 3600, 4),
                   "history": history, "test_metrics": test_metrics}, f, indent=2)


if __name__ == "__main__":      # run main() only when executed as a script (not when imported)
    main()
