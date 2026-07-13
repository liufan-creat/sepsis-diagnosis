"""
BERT-style Transformer classifier for sepsis diagnosis prediction.

Architecture:
  tok_embed (Embedding) -> [TransformerEncoder] x encoder_num -> LayerNorm -> Linear -> GELU -> Dropout -> CLS token -> Linear(class_num)

Built to match the inline Bert class from mimic_sepsis_diag.py exactly.
"""

import torch
import torch.nn as nn


class BertClassifier(nn.Module):
    """Multi-branch transformer classifier for diagnostic text."""

    def __init__(self, vocab_size, label_dim=128, hidden_size=256,
                 num_layers=6, num_heads=8, class_num=2, encoder_num=6,
                 dropout=0.1):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, label_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=label_dim, nhead=num_heads,
            activation='gelu', norm_first=True, batch_first=True
        )
        module_list = [nn.TransformerEncoder(encoder_layer, num_layers)
                       for _ in range(encoder_num)]
        self.module_list = nn.ModuleList(module_list)

        self.linear = nn.Linear(label_dim, hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(label_dim)
        self.linear_cls = nn.Linear(hidden_size, class_num)

    def forward(self, input_ids, attention_mask, return_embedding=False):
        word_embeding = self.tok_embed(input_ids)

        for module in self.module_list:
            output = module(word_embeding, src_key_padding_mask=attention_mask)

        output1 = self.layer_norm(output)
        output = self.linear(output1)
        output = self.activation(output)
        output = self.dropout(output)
        output = output[:, 0, :]
        output = self.linear_cls(output)

        if return_embedding:
            return output, output1
        return output
