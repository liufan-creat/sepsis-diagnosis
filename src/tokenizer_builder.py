"""
Tokenizer builder — creates HuggingFace PreTrainedTokenizerFast from raw text corpora.
"""

import json
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from config import SPECIAL_TOKENS


def build_and_save_tokenizer(word_list, vocab, tokenizer_path):
    """
    Build a WordPiece tokenizer from a list of words and save.

    Args:
        word_list: list[str] — words to add to vocabulary
        vocab: list[str] — existing vocabulary words already in tokenizer
        tokenizer_path: str or Path — output .json path

    Returns:
        PreTrainedTokenizerFast instance
    """
    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tokenizer.add_special_tokens(SPECIAL_TOKENS)
    tokenizer.add_tokens(word_list)
    tokenizer.save(str(tokenizer_path))

    fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path))
    fast_tokenizer.add_special_tokens({
        "pad_token": "[PAD]", "unk_token": "[UNK]",
        "cls_token": "[CLS]", "sep_token": "[SEP]",
        "mask_token": "[MASK]"
    })
    return fast_tokenizer


def load_tokenizer(tokenizer_path):
    """Load a PreTrainedTokenizerFast from saved .json."""
    fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path))
    fast_tokenizer.add_special_tokens({
        "pad_token": "[PAD]", "unk_token": "[UNK]",
        "cls_token": "[CLS]", "sep_token": "[SEP]",
        "mask_token": "[MASK]"
    })
    return fast_tokenizer


def build_shap_tokenizer(fast_tokenizer):
    """
    Prepare tokenizer for SHAP — assign the WordPiece model reference.
    The SHAP TextMasker needs fast_tokenizer._tokenizer.model set.
    """
    vocab = fast_tokenizer.get_vocab()
    fast_tokenizer._tokenizer.model = models.WordPiece(
        vocab=vocab, unk_token="[UNK]"
    )
    return fast_tokenizer
