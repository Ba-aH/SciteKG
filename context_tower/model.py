"""
context_tower/model.py
----------------------
ContextTower: encodes a citing passage (context text) into a fixed-dim
L2-normalised embedding that lives in the same space as PaperTower's output.

Architecture:
    SciBERT (fine-tuned)           — allenai/scibert_scivocab_uncased
    CLS pooling                    — take hidden state of [CLS] token
    Linear(hidden_size, embed_dim) — project into shared embedding space
    LayerNorm(embed_dim)           — stabilise activations
    L2 normalise                   — unit sphere, consistent with PaperTower

Input:
    input_ids      : LongTensor  [B, seq_len]
    attention_mask : LongTensor  [B, seq_len]
    (token_type_ids are not needed for SciBERT single-sequence inputs)

Output:
    Tensor[B, embed_dim], L2-normalised.

Notes:
- SciBERT hidden_size = 768, matching nfeat in PaperTower.
- embed_dim must match the value used in PaperTower (default 256).
- All SciBERT weights are trainable (fine-tuned end-to-end).
- A lower learning rate for SciBERT vs the projection head is recommended
  in train.py (e.g. 1e-5 for SciBERT, 1e-3 for proj layers).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class ContextTower(nn.Module):
    """
    Parameters
    ----------
    embed_dim : int
        Output embedding dimension. Must match PaperTower's embed_dim.
    scibert_model_name : str
        HuggingFace model identifier for SciBERT.
    dropout : float
        Dropout applied after CLS pooling, before the projection layer.
    """


    def __init__(
        self,
        embed_dim: int,
        scibert_model_name: str,
        dropout: float,
    ):
        super().__init__()

        self.embed_dim = embed_dim

        # --- SciBERT encoder (all weights trainable) ---
        self.scibert = AutoModel.from_pretrained(scibert_model_name)

        # --- Projection head ---
        self.dropout = nn.Dropout(dropout)
        self.proj    = nn.Linear(768, embed_dim) # 768 is the hidden size of SciBERT,it's a property of the model, not a hyperparameter.
                                                 # HIDDEN: it's the dimension of the vector that represents each token after passing through the transformer layers.
        self.norm    = nn.LayerNorm(embed_dim)

        self._init_projection()

    # ------------------------------------------------------------------
    def _init_projection(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.proj.weight, gain=gain)
        nn.init.zeros_(self.proj.bias)

    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,       # [B, seq_len] tokenized citation context
        attention_mask: torch.Tensor,  # [B, seq_len] 1 for real tokens, 0 for padding
    ) -> torch.Tensor:

        # Run SciBERT — each of the seq_len tokens gets a 768-dim contextual representation
        outputs = self.scibert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Take only the [CLS] token (position 0) as the sentence-level embedding → [B, 768]
        # [CLS] is a special token prepended to every input, trained to summarize the whole sequence
        cls_emb = outputs.last_hidden_state[:, 0, :] # Here we take the [CLS] token's embedding from the last hidden state of the model.

        # Project into shared embedding space so both towers output the same dimension
        x = self.dropout(cls_emb)  # randomly zero some dimensions to prevent overfitting
        x = self.proj(x)           # PASS THE CLS EMBEDDING THROUGH A LINEAR LAYER TO CHANGE ITS DIMENSION FROM 768 TO embed_dim if needed
        x = self.norm(x)           # normalize the projected embedding to have zero mean and unit variance across the batch, which can help with training stability.

        # L2 normalise → place embedding on unit sphere so dot product = cosine similarity
        return F.normalize(x, p=2, dim=-1)

    # ------------------------------------------------------------------
    def get_param_groups(self, lr_scibert: float, lr_head: float) -> list:
        """
        Returns two parameter groups with different learning rates,
        ready to pass directly to an optimiser.

        Usage in train.py:
            optimizer = torch.optim.Adam(
                model.get_param_groups(lr_scibert=1e-5, lr_head=1e-3)
            )
        """
        return [
            {"params": self.scibert.parameters(), "lr": lr_scibert},
            {"params": list(self.proj.parameters())
                     + list(self.norm.parameters()),  "lr": lr_head},
        ]


