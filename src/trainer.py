"""
Training loop for the BERT classifier.
Matches the original scripts: random mini-batch sampling per step.
"""

import torch
import numpy as np


def train_model(model, train_data, tokenizer, max_length, device,
                loss_fn, optimizer, max_steps=5000, batch_size=64, verbose=True):
    """
    Train with per-step random sampling (matching original scripts).

    Each step randomly samples `batch_size` rows from train_data,
    tokenizes them, and performs one gradient update.

    Args:
        model: BertClassifier instance
        train_data: DataFrame with 'input' and 'hospital_expire_flag' columns
        tokenizer: PreTrainedTokenizerFast
        max_length: int, max sequence length for padding
        device: torch.device
        loss_fn: nn.CrossEntropyLoss
        optimizer: torch optimizer
        max_steps: number of training steps
        batch_size: sample size per step
        verbose: print progress every 500 steps

    Returns:
        list of per-step loss values
    """
    model.train()
    ce_loss = []

    for step in range(max_steps):
        data_train = train_data.sample(n=min(batch_size, len(train_data)), random_state=None)
        label = torch.tensor(data_train["hospital_expire_flag"].values, dtype=torch.long)

        inputs = tokenizer(data_train["input"].astype(str).tolist(),
                           padding="max_length", max_length=max_length,
                           return_tensors="pt", return_offsets_mapping=False)

        encodings = inputs["input_ids"].to(device)
        pad_mask = inputs["attention_mask"].to(torch.bool) == False
        pad_mask = pad_mask.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        output = model(encodings, attention_mask=pad_mask)
        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()

        ce_loss.append(loss.item())

        if verbose and (step % 500 == 0 or step == max_steps - 1):
            print(f"  Step {step}/{max_steps}, Loss: {loss.item():.4f}")

    return ce_loss


def evaluate_model(model, test_data, tokenizer, max_length, device, return_embedding=False):
    """
    Run inference on test data.

    Returns:
        predictions: np.array of class predictions
        logits: np.array of raw logits (if return_embedding)
        embedding: np.array of pre-pooling embeddings at index 0 (if return_embedding)
    """
    model.eval()
    texts = test_data["input"].astype(str).tolist()

    inputs = tokenizer(texts, padding="max_length", max_length=max_length,
                       return_tensors="pt", return_offsets_mapping=False)

    encodings = inputs["input_ids"].to(device)
    pad_mask = inputs["attention_mask"].to(torch.bool) == False
    pad_mask = pad_mask.to(device)

    with torch.no_grad():
        if return_embedding:
            output, emb = model(encodings, attention_mask=pad_mask, return_embedding=True)
            predictions = torch.argmax(output, dim=-1).cpu().numpy()
            logits = output.cpu().numpy()
            embeddings = emb[:, 0, :].cpu().numpy()
            return predictions, logits, embeddings
        else:
            output = model(encodings, attention_mask=pad_mask)
            predictions = torch.argmax(output, dim=-1).cpu().numpy()
            return predictions
