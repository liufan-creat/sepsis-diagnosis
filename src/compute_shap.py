#!/usr/bin/env python3
"""
Compute SHAP permutation importance values for model predictions.

Fixed: unk_token fix + auto-detect encoder_num from checkpoint.

Usage:
    python compute_shap.py --checkpoint outputs/BERT-36L-CSL.pth \\
                           --tokenizer outputs/sepsis_diagnoses_tokenizer.json \\
                           --texts outputs/data_t_diag.csv --text-col input \\
                           --n-samples 30 --output shap_results.csv
"""

import argparse
import pandas as pd
import torch
from pathlib import Path
from tokenizers.models import WordPiece

from bert_classifier import BertClassifier
from tokenizer_builder import load_tokenizer, build_shap_tokenizer
from config import LABEL_DIM, HIDDEN_SIZE, NUM_HEADS, CLASS_NUM, DROPOUT


def compute_shap_values(model, tokenizer, texts, device, n_samples=50):
    """Compute SHAP permutation values for a subset of texts.

    Returns:
        DataFrame where each row is a dict {token: shap_value}
    """
    tokenizer = build_shap_tokenizer(tokenizer)

    def predict_fn(sample):
        inputs = tokenizer(
            sample.astype(str).tolist(),
            padding="max_length", max_length=45,
            return_tensors="pt", return_offsets_mapping=False,
        )
        enc = inputs["input_ids"].to(device)
        mask = (inputs["attention_mask"].to(torch.bool) == False).to(device)
        with torch.no_grad():
            out = model(enc, attention_mask=mask)
        return torch.argmax(out, dim=-1).cpu().numpy()

    import shap
    masker = shap.maskers.Text(tokenizer, mask_token="[MASK]")
    explainer = shap.Explainer(predict_fn, masker, algorithm="permutation")

    texts_series = texts if isinstance(texts, pd.Series) else texts["input"]
    texts_sample = texts_series.head(n_samples)
    shap_vals = explainer(texts_sample)
    values = shap_vals.abs.values

    shape_data_list = []
    for i in range(len(values)):
        pro_tokens = texts_sample.iloc[i].split()[1:]  # skip [CLS]
        val_arr = values[i][1:]  # skip [CLS] position
        min_len = min(len(pro_tokens), len(val_arr))
        shape_dict = dict(zip(pro_tokens[:min_len], val_arr[:min_len]))
        shape_dict = {k: v for k, v in shape_dict.items() if v != 0}
        shape_data_list.append(shape_dict)

    return pd.DataFrame(shape_data_list)


def main():
    parser = argparse.ArgumentParser(description="Compute SHAP permutation values")
    parser.add_argument("--checkpoint", "-c", required=True, help="Model checkpoint .pth")
    parser.add_argument("--tokenizer", "-t", required=True, help="Tokenizer JSON path")
    parser.add_argument("--texts", "-x", required=True, help="CSV with 'input' column")
    parser.add_argument("--text-col", default="input", help="Column name for text (default: input)")
    parser.add_argument("--n-samples", "-n", type=int, default=30, help="Number of samples to analyze")
    from config import OUTPUT_DIR
    parser.add_argument("--output-dir", "-d", default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--output", "-o", default=None, help="Output CSV path (overrides --output-dir)")
    parser.add_argument("--device", default=None, help="CUDA device (e.g. cuda:0)")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    # Load tokenizer (with unk_token fix)
    tokenizer_obj = load_tokenizer(args.tokenizer)
    vocab = tokenizer_obj.get_vocab()
    tokenizer_obj._tokenizer.model = WordPiece(vocab=vocab, unk_token="[UNK]")

    # Auto-detect encoder_num from checkpoint
    ckpt_data = torch.load(args.checkpoint, map_location=device, weights_only=False)
    vocab_size = ckpt_data["model_state_dict"]["tok_embed.weight"].shape[0]
    branch_ids = set(
        k.split(".")[1] for k in ckpt_data["model_state_dict"].keys()
        if k.startswith("module_list.")
    )
    encoder_num = len(branch_ids)

    model = BertClassifier(
        vocab_size=vocab_size,
        label_dim=LABEL_DIM, hidden_size=HIDDEN_SIZE,
        num_layers=6, num_heads=NUM_HEADS, class_num=CLASS_NUM,
        encoder_num=encoder_num, dropout=DROPOUT,
    ).to(device)
    model.load_state_dict(ckpt_data["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint} (encoder_num={encoder_num})")

    # Load texts
    df = pd.read_csv(args.texts)
    texts = df[args.text_col]

    # Compute SHAP
    print(f"Computing SHAP on {len(texts)} texts (sampling {args.n_samples})...")
    shap_df = compute_shap_values(model, tokenizer_obj, texts, device, n_samples=args.n_samples)
    out_path = args.output or f"{args.output_dir}/shap_results.csv"
    shap_df.to_csv(out_path, index=False)
    print(f"SHAP results saved → {out_path}")


if __name__ == "__main__":
    main()
