"""
whole_group_sampler.py
----------------------
Batch sampler that keeps every record of one citation-context group TOGETHER in the same batch.

Where to put it:
    ~/HGNN/shared/data_prep/whole_group_sampler.py      (next to dataset.py and group_aware_sampler.py)

Why it exists:
    all_contexts.json has one record per (context, cited paper). A marker that cites several papers
    (citation_type "multiple") therefore appears as several records that share the same text and the
    same context_group_id. Your GroupAwareBatchSampler does the OPPOSITE of this file: it keeps such
    sibling records in different batches, because in the two-tower model siblings would be false
    negatives of each other. The multi-slot model needs the reverse: all papers cited by one marker
    must travel together, because they become ONE training example with several gold papers.

How it is used (see train_cite_lm_slots.py):
    sampler = WholeGroupBatchSampler(group_ids, groups_per_batch=64, seed=42, shuffle=True)
    loader  = DataLoader(dataset, batch_sampler=sampler, collate_fn=lcr_collate_fn)
    for epoch in range(...):
        sampler.set_epoch(epoch)          # new group order every epoch (training only)
        for batch in loader: ...

It has the same interface as GroupAwareBatchSampler: set_epoch(), __len__ and __iter__ yielding
lists of record indices. It has no dependency on torch.
"""

import math                        # math.ceil for the number of batches
import random                      # random.Random for a reproducible, epoch-dependent shuffle


# ---------------------------------------------------------------------------
# WholeGroupBatchSampler (class)
#
# Builds batches that contain COMPLETE context groups. A batch holds `groups_per_batch` groups, so
# it holds a bit more than that many records (about 1.17 records per group in your data).
#
# Arguments:
#   group_ids        : list with the context_group_id of every record, in dataset order
#                      (e.g. [r.context_group_id for r in dataset.records])
#   groups_per_batch : how many groups form one batch
#   seed             : base seed for the shuffling (the epoch number is added to it)
#   shuffle          : True for training (new order each epoch), False for validation / test
#                      (groups are then visited in order of first appearance, deterministically)
# ---------------------------------------------------------------------------
class WholeGroupBatchSampler:
    def __init__(self, group_ids, groups_per_batch, seed, shuffle):
        by_group = {}                                         # group id -> list of record indices
        for idx, gid in enumerate(group_ids):                 # walk over all records in dataset order
            by_group.setdefault(gid, []).append(idx)          # append the record to its group
        self.groups = list(by_group.values())                 # list of index-lists (order of first appearance)
        self.groups_per_batch = groups_per_batch              # how many groups form one batch
        self.seed = seed                                      # base seed for the shuffling
        self.shuffle = shuffle                                # True for training, False for val/test
        self.epoch = 0                                        # changes the shuffle every epoch

    # Called by the training loop at the start of every epoch so the group order changes.
    def set_epoch(self, epoch):
        self.epoch = epoch

    # Number of batches per epoch (used by tqdm and DataLoader).
    def __len__(self):
        return math.ceil(len(self.groups) / self.groups_per_batch)

    # Yields one list of record indices per batch; DataLoader fetches exactly these records.
    def __iter__(self):
        order = list(range(len(self.groups)))                 # group positions 0..n_groups-1
        if self.shuffle:                                      # training: new order each epoch, reproducible
            random.Random(self.seed + self.epoch).shuffle(order)
        for s in range(0, len(order), self.groups_per_batch): # slice the order into batches of groups
            batch = []                                        # record indices of this batch
            for gi in order[s : s + self.groups_per_batch]:   # for each group in the slice...
                batch.extend(self.groups[gi])                 # ...add ALL of its records
            yield batch
